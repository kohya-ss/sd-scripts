"""Probe which Anima DiT block(s) make the best ref_block for the LLLite v3 semantic trunk.

学習なしで、凍結 DiT の各ブロックの hidden states が v3 semantic trunk の条件源として
どれだけ適しているかを、手持ちの学習ペア (条件画像 / ターゲット画像) から測る。

各ブロック k について 2 系統の指標を測定する:

  1. 意味分離性 (probe AUROC):
     条件画像の block-k 特徴だけから「変更された領域 (latent diff の擬似マスク)」を
     線形分類できるか。学習ペアの一部で ridge 回帰の線形 probe を閉形式で解き、
     held-out ペアの AUROC を報告する。ゲートが必要とする「どこを書き換えるか」の
     判断材料がその深さに存在するかの直接測定。

  2. コピー信号の保存性 (preservation / response):
     変更されていない領域で、条件画像とターゲット画像の同一位置の特徴が一致するか
     (matched cosine - shuffled cosine)。値パスが必要とする位置対応の外観情報が
     残っているかの測定。変更領域での同指標 (response、低いほど良い) との差
     (contrast) は「画像が同じ場所では一致し、違う場所では乖離する」度合い。

期待される読み方: 意味分離性は深さとともに上がりどこかで飽和・劣化し (出力特化と
分布外ドリフトの蓄積)、保存性は深さとともに落ちる。両者が高い交点付近が single
ref_block の sweet spot。2 ブロック concat なら「保存性最良の浅めブロック +
意味分離性最良の深めブロック」が候補になる。

参照フォワードは学習時と同じ規約 (zero context / 固定 ref_timestep / no_grad)。

Run (example):
    python tools/dev/probe_anima_ref_blocks.py \
      --dit <anima.safetensors> --vae <qwen_image_vae.safetensors> \
      --image_dir <target images> --cond_dir <condition images> \
      --image_size 1024 1024 --num_pairs 32 --device cuda
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# repo root on sys.path (this file lives in tools/dev/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from library import anima_train_utils, anima_utils
from library import attention as attention_lib
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Anima DiT blocks for LLLite v3 ref_block selection")
    parser.add_argument("--dit", type=str, required=True, help="Anima DiT checkpoint path")
    parser.add_argument("--vae", type=str, required=True, help="Qwen-Image VAE path")
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--qwen_image_vae_2d", action="store_true")
    parser.add_argument("--image_dir", type=str, required=True, help="target images (image_dir of the dataset)")
    parser.add_argument("--cond_dir", type=str, required=True, help="condition images (conditioning_data_dir)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="height width (multiple of 16)")
    parser.add_argument("--num_pairs", type=int, default=32, help="number of pairs to sample")
    parser.add_argument("--eval_every", type=int, default=4, help="every N-th pair goes to the probe eval split")
    parser.add_argument("--ref_timestep", type=float, default=0.0, help="reference forward timestep ([0,1] scale)")
    parser.add_argument(
        "--changed_quantile", type=float, default=0.75,
        help="tokens with latent diff above this per-image quantile are labeled 'changed' (default: top 25%%)",
    )
    parser.add_argument(
        "--unchanged_quantile", type=float, default=0.50,
        help="tokens with latent diff below this per-image quantile are labeled 'unchanged' (default: bottom 50%%)",
    )
    parser.add_argument("--tokens_per_class", type=int, default=1024, help="max tokens per class per pair for the probe")
    parser.add_argument("--ridge_lambda", type=float, default=10.0, help="ridge regularization for the linear probe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers", "sdpa"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    h, w = args.image_size
    assert h % 16 == 0 and w % 16 == 0, f"--image_size must be multiples of 16, got {h}x{w}"
    assert 0.0 < args.unchanged_quantile < args.changed_quantile < 1.0
    return args


def list_pairs(image_dir: str, cond_dir: str) -> list:
    def stems(d):
        out = {}
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                out[stem] = os.path.join(d, f)
        return out

    tgt = stems(image_dir)
    cnd = stems(cond_dir)
    common = sorted(set(tgt) & set(cnd))
    return [(cnd[s], tgt[s]) for s in common]


def load_image(path: str, height: int, width: int, device, dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def all_block_hidden_states(dit, latent: torch.Tensor, ref_timestep: float) -> list:
    """encode_reference_hidden_states と同じ規約 (zero context / 固定 t) で全ブロックの
    hidden states を取り、[(S, D)] * num_blocks (fp32, LayerNorm 正規化済み) を返す。

    LayerNorm(D) を掛けるのは semantic trunk の ln_in と同じ前処理で probe するため。
    """
    x5 = latent.unsqueeze(2)  # (1, 16, 1, h, w)
    bsz = x5.shape[0]
    h_lat, w_lat = x5.shape[-2], x5.shape[-1]
    padding_mask = torch.zeros(bsz, 1, h_lat, w_lat, device=latent.device, dtype=latent.dtype)

    x, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x5, fps=None, padding_mask=padding_mask)

    t = torch.full((bsz, 1), float(ref_timestep), device=latent.device, dtype=latent.dtype)
    t_emb, adaln_lora = dit.t_embedder(t)
    t_emb = dit.t_embedding_norm(t_emb)

    ctx_dim = dit.blocks[0].cross_attn.context_dim
    context = torch.zeros(bsz, 1, ctx_dim, device=latent.device, dtype=latent.dtype)
    attn_params = attention_lib.AttentionParams.create_attention_params(dit.attn_mode, dit.split_attn)
    use_fp32 = x.dtype == torch.float16

    feats = []
    for block in dit.blocks:
        x = block(
            x, t_emb, context, attn_params, use_fp32,
            rope_emb_L_1_1_D=rope_emb, adaln_lora_B_T_3D=adaln_lora,
            extra_per_block_pos_emb=extra_pos_emb,
        )
        f = x.reshape(-1, x.shape[-1]).float()  # (S, D)
        feats.append(F.layer_norm(f, (f.shape[-1],)))
    return feats


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """rank-based AUROC. labels in {0, 1}."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[labels.bool()].sum().item()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    pairs = list_pairs(args.image_dir, args.cond_dir)
    if not pairs:
        raise SystemExit(f"no filename-matched pairs between {args.image_dir} and {args.cond_dir}")
    random.shuffle(pairs)
    pairs = pairs[: args.num_pairs]
    logger.info(f"{len(pairs)} pairs sampled")

    logger.info("Loading VAE...")
    vae = anima_train_utils.load_qwen_image_vae(args, device=device, disable_mmap=True)
    vae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(device, args.dit, args.attn_mode, False, device, dit_weight_dtype=dtype)
    dit.to(device).eval().requires_grad_(False)
    num_blocks = len(dit.blocks)
    d_model = dit.model_channels
    logger.info(f"num_blocks={num_blocks}, model_channels={d_model}")

    height, width = args.image_size
    tok_h, tok_w = height // 16, width // 16
    n_tokens = tok_h * tok_w

    # probe 用のストリーミング積算 (train split): ブロックごとに Gram (D+1)^2 と X^T y
    gram = torch.zeros(num_blocks, d_model + 1, d_model + 1, device=device, dtype=torch.float64)
    xty = torch.zeros(num_blocks, d_model + 1, device=device, dtype=torch.float64)
    # eval split のトークンは CPU に保持
    eval_store = [[] for _ in range(num_blocks)]  # per block: list of (feats fp16 cpu, labels cpu)

    # 保存性/応答性のストリーミング平均 (全ペア)
    pres_sum = torch.zeros(num_blocks, dtype=torch.float64)   # unchanged: matched - shuffled
    resp_sum = torch.zeros(num_blocks, dtype=torch.float64)   # changed:   matched - shuffled
    pres_cnt = 0

    changed_frac_sum = 0.0
    n_train_tokens = 0

    for pair_idx, (cond_path, tgt_path) in enumerate(pairs):
        is_eval = (pair_idx % args.eval_every) == args.eval_every - 1

        cond_img = load_image(cond_path, height, width, device, dtype)
        tgt_img = load_image(tgt_path, height, width, device, dtype)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            lat_c = vae.encode_pixels_to_latents(cond_img.to(vae.dtype)).to(dtype)
            lat_t = vae.encode_pixels_to_latents(tgt_img.to(vae.dtype)).to(dtype)
        lat_c, lat_t = lat_c.squeeze(2) if lat_c.dim() == 5 else lat_c, lat_t.squeeze(2) if lat_t.dim() == 5 else lat_t

        # 擬似マスク: latent diff の channel L2 -> token grid -> 分位点ラベル
        diff = (lat_t.float() - lat_c.float()).pow(2).sum(dim=1, keepdim=True).sqrt()  # (1,1,h,w)
        diff_tok = F.avg_pool2d(diff, 2).reshape(-1)  # (S,)
        q_hi = torch.quantile(diff_tok, args.changed_quantile)
        q_lo = torch.quantile(diff_tok, args.unchanged_quantile)
        changed = diff_tok >= q_hi
        unchanged = diff_tok <= q_lo
        changed_frac_sum += changed.float().mean().item()

        feats_c = all_block_hidden_states(dit, lat_c, args.ref_timestep)
        feats_t = all_block_hidden_states(dit, lat_t, args.ref_timestep)

        idx_ch = changed.nonzero(as_tuple=True)[0]
        idx_un = unchanged.nonzero(as_tuple=True)[0]
        # probe 用にクラスごと上限までサブサンプル
        sub_ch = idx_ch[torch.randperm(len(idx_ch), device=device)[: args.tokens_per_class]]
        sub_un = idx_un[torch.randperm(len(idx_un), device=device)[: args.tokens_per_class]]
        sub_idx = torch.cat([sub_ch, sub_un])
        sub_lab = torch.cat([
            torch.ones(len(sub_ch), device=device), torch.zeros(len(sub_un), device=device)
        ])

        # 保存性の shuffled baseline 用置換 (unchanged / changed 内で独立に)
        perm_un = idx_un[torch.randperm(len(idx_un), device=device)]
        perm_ch = idx_ch[torch.randperm(len(idx_ch), device=device)]

        for k in range(num_blocks):
            fc, ft = feats_c[k], feats_t[k]  # (S, D) fp32, layer-normed

            # --- probe (条件画像の特徴のみ使用) ---
            xs = fc[sub_idx]  # (n, D)
            xs1 = torch.cat([xs, torch.ones(len(xs), 1, device=device)], dim=1).double()
            if is_eval:
                eval_store[k].append((xs.half().cpu(), sub_lab.cpu()))
            else:
                gram[k] += xs1.T @ xs1
                xty[k] += xs1.T @ sub_lab.double()

            # --- preservation / response (cond vs target, matched - shuffled cosine) ---
            fcn = F.normalize(fc, dim=-1)
            ftn = F.normalize(ft, dim=-1)
            cos_un = (fcn[idx_un] * ftn[idx_un]).sum(-1).mean()
            cos_un_shuf = (fcn[idx_un] * ftn[perm_un]).sum(-1).mean()
            cos_ch = (fcn[idx_ch] * ftn[idx_ch]).sum(-1).mean()
            cos_ch_shuf = (fcn[idx_ch] * ftn[perm_ch]).sum(-1).mean()
            pres_sum[k] += (cos_un - cos_un_shuf).double().cpu()
            resp_sum[k] += (cos_ch - cos_ch_shuf).double().cpu()

        if not is_eval:
            n_train_tokens += len(sub_idx)
        pres_cnt += 1
        del feats_c, feats_t
        logger.info(
            f"pair {pair_idx + 1}/{len(pairs)} ({'eval' if is_eval else 'train'}): "
            f"{os.path.basename(cond_path)} changed={changed.float().mean().item():.2%}"
        )

    # --- 線形 probe を閉形式で解き、eval AUROC を計算 ---
    logger.info(f"solving ridge probes (lambda={args.ridge_lambda}, train tokens={n_train_tokens})...")
    aurocs = []
    eye = torch.eye(d_model + 1, device=device, dtype=torch.float64) * args.ridge_lambda
    for k in range(num_blocks):
        if not eval_store[k]:
            aurocs.append(float("nan"))
            continue
        w = torch.linalg.solve(gram[k] + eye, xty[k])  # (D+1,)
        xs = torch.cat([f for f, _ in eval_store[k]]).float().to(device)
        ys = torch.cat([l for _, l in eval_store[k]]).to(device)
        scores = xs @ w[:-1].float() + w[-1].float()
        aurocs.append(auroc(scores.cpu(), ys.cpu()))

    pres = (pres_sum / pres_cnt).tolist()
    resp = (resp_sum / pres_cnt).tolist()

    # --- 報告 ---
    print()
    print(f"pairs={len(pairs)} (eval every {args.eval_every}), image={height}x{width}, "
          f"tokens/image={n_tokens}, ref_timestep={args.ref_timestep}, "
          f"mean changed fraction={changed_frac_sum / len(pairs):.2%}")
    print()
    print("block | probe AUROC | preservation | response | contrast")
    print("      | (semantic)  | (unchanged)  | (changed)| (pres - resp)")
    print("------+-------------+--------------+----------+---------")
    for k in range(num_blocks):
        contrast = pres[k] - resp[k]
        print(f"  {k:3d} |    {aurocs[k]:.4f}   |    {pres[k]:+.4f}   | {resp[k]:+.4f}  | {contrast:+.4f}")

    # --- 推奨 ---
    valid = [k for k in range(num_blocks) if not np.isnan(aurocs[k])]
    a = np.array([aurocs[k] for k in valid])
    p = np.array([pres[k] for k in valid])

    def _norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    combined = _norm(a) + _norm(p)
    top_single = [valid[i] for i in np.argsort(-combined)[:3]]
    best_sem = valid[int(np.argmax(a))]
    shallow_half = [i for i, k in enumerate(valid) if k < num_blocks // 2]
    best_app = valid[shallow_half[int(np.argmax(p[shallow_half]))]] if shallow_half else valid[int(np.argmax(p))]
    print()
    print(f"suggested single ref_block (normalized AUROC + preservation): {top_single}")
    print(f"suggested dual-block concat: shallow={best_app} (best preservation in the shallow half), "
          f"deep={best_sem} (best probe AUROC)")
    print()
    print("Reading guide: AUROC = can a linear probe locate the change region from the cond features")
    print("alone (gate material). preservation = position-matched appearance signal in unchanged")
    print("regions (value-path material). Expect AUROC to rise then degrade with depth, preservation")
    print("to decay; the sweet spot is where both are high.")


if __name__ == "__main__":
    main()
