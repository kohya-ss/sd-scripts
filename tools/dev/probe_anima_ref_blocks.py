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

参照フォワードは学習時と同じ規約 (固定 ref_timestep / no_grad)。context は --context で選択:

  - zero             : ゼロ context (v3 MVP の既定。cross-attn 出力が厳密に 0 = テキスト非依存)
  - uncond           : 空文字列プロンプトを text encoder + LLM Adapter に通した context
                       (分布内の無情報 context。caption dropout でモデルが学習中に見ている条件)
  - caption          : ターゲット画像のキャプション (生成プロンプト) を通した context
                       = プロンプト条件付き参照 (references §6.4) の事前見積もり
  - caption_shuffled : 別ペアのキャプションを割り当てた対照条件 (1 ローテーション。
                       ペア順は seed 付きシャッフル済みなのでランダム対応と等価)

カンマ区切りで複数指定すると同一ペア・同一トークンサブサンプルで並走し、モード間の
AUROC / preservation を対比較する。分解の読み方:

  uncond - zero               = 分布内化の効果 (テキスト情報ゼロのまま)
  caption_shuffled - uncond   = 「もっともらしい自然文なら何でも良い」効果
  caption - caption_shuffled  = 画像とプロンプトの対応関係の効果 (矛盾検出そのもの)。
                                ここが正なら instruct-pix2pix 型配線の直接的な根拠になる。

注意: モード数に比例して DiT フォワード回数と ridge 積算メモリ (~1GB/モード @ D=2048) が
増える。VRAM が苦しければモードを分けて実行してよい (同一 seed ならペア選択は一致する)。

Run (example):
    python tools/dev/probe_anima_ref_blocks.py \
      --dit <anima.safetensors> --vae <qwen_image_vae.safetensors> \
      --image_dir <target images> --cond_dir <condition images> \
      --image_size 1024 1024 --num_pairs 32 --device cuda \
      --context zero,uncond,caption,caption_shuffled --text_encoder <qwen3 dir or safetensors>
"""

import argparse
import gc
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
        "--context", type=str, default="zero",
        help="comma-separated reference-forward context modes: zero | uncond | caption | caption_shuffled "
             "(see module docstring)",
    )
    parser.add_argument(
        "--text_encoder", type=str, default=None,
        help="Qwen3 Text Encoder path (dir or safetensors); required for --context uncond/caption",
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".txt",
        help="caption file extension, looked up next to each target image (--context caption)",
    )
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

    args.context_modes = [m.strip() for m in args.context.split(",") if m.strip()]
    valid_modes = ("zero", "uncond", "caption", "caption_shuffled")
    assert args.context_modes, "--context must specify at least one mode"
    assert all(m in valid_modes for m in args.context_modes), (
        f"--context modes must be in {valid_modes}, got {args.context_modes}"
    )
    assert len(set(args.context_modes)) == len(args.context_modes), f"duplicate --context modes: {args.context_modes}"
    if any(m != "zero" for m in args.context_modes):
        assert args.text_encoder, "--text_encoder is required for --context uncond/caption"
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
def all_block_hidden_states(dit, latent: torch.Tensor, ref_timestep: float, context: torch.Tensor = None) -> list:
    """encode_reference_hidden_states と同じ規約 (固定 t) で全ブロックの
    hidden states を取り、[(S, D)] * num_blocks (fp32, LayerNorm 正規化済み) を返す。

    context=None ならゼロ context (v3 MVP の既定)。それ以外は LLM Adapter 通過済みの
    (1, L, ctx_dim) を渡す (padding 位置はゼロ潰し済みであること)。
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

    if context is None:
        ctx_dim = dit.blocks[0].cross_attn.context_dim
        context = torch.zeros(bsz, 1, ctx_dim, device=latent.device, dtype=latent.dtype)
    else:
        context = context.to(device=latent.device, dtype=latent.dtype)
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


@torch.no_grad()
def encode_prompts(text_encoder_path: str, prompts: list, device, dtype) -> dict:
    """Qwen3 text encoder で prompt 群をエンコードし {prompt: [prompt_embeds, attn_mask,
    t5_input_ids, t5_attn_mask] (CPU)} を返す。エンコード後にモデルは解放する。

    LLM Adapter は DiT 側にあるため、ここでは Qwen3 空間の埋め込みまで
    (学習・推論と同じ AnimaTokenizeStrategy / AnimaTextEncodingStrategy を使用)。
    """
    from library import strategy_anima

    logger.info(f"Loading Qwen3 text encoder from {text_encoder_path} ...")
    text_encoder, tokenizer = anima_utils.load_qwen3_text_encoder(text_encoder_path, dtype=dtype, device=device)
    text_encoder.eval().requires_grad_(False)
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=tokenizer, t5_tokenizer=None, qwen3_max_length=512, t5_max_length=512
    )
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    out = {}
    for prompt in dict.fromkeys(prompts):  # unique, order-preserving
        tokens = tokenize_strategy.tokenize(prompt)
        embeds = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
        out[prompt] = [t.cpu() for t in embeds]
    logger.info(f"encoded {len(out)} unique prompt(s)")

    del text_encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def build_llm_adapter_context(dit, embeds: list, device, dtype) -> torch.Tensor:
    """encode_prompts の出力 1 件を DiT の LLM Adapter に通し、cross-attn 用 context
    (1, L_t5, ctx_dim) を返す。padding 位置は _preprocess_text_embeds 内でゼロ潰しされる。"""
    prompt_embeds = embeds[0].to(device=device, dtype=dtype)
    attn_mask = embeds[1].to(device)
    t5_input_ids = embeds[2].to(device=device, dtype=torch.long)
    t5_attn_mask = embeds[3].to(device)
    return dit._preprocess_text_embeds(
        prompt_embeds, t5_input_ids, target_attention_mask=t5_attn_mask, source_attention_mask=attn_mask
    )


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

    modes = args.context_modes

    # --- captions (target 画像の生成プロンプト) ---
    captions = None
    if "caption" in modes or "caption_shuffled" in modes:
        if "caption_shuffled" in modes:
            assert len(pairs) >= 2, "--context caption_shuffled requires at least 2 pairs"
        captions, missing = [], []
        for _, tgt_path in pairs:
            cap_path = os.path.splitext(tgt_path)[0] + args.caption_extension
            if not os.path.exists(cap_path):
                missing.append(cap_path)
                continue
            with open(cap_path, "r", encoding="utf-8") as f:
                captions.append(f.read().strip())
        if missing:
            raise SystemExit(
                f"--context caption: {len(missing)} caption file(s) missing next to target images, e.g. {missing[:3]}"
            )

    # --- text encoder (Qwen3 空間まで先にエンコードして解放。LLM Adapter は DiT ロード後) ---
    prompt_embeds_map = {}
    if any(m != "zero" for m in modes):
        to_encode = []
        if "uncond" in modes:
            to_encode.append("")  # 空文字列 uncond (学習の caption dropout / 推論の negative "" と同じ)
        if captions is not None:
            to_encode.extend(captions)
        prompt_embeds_map = encode_prompts(args.text_encoder, to_encode, device, dtype)

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

    # --- context の準備 (uncond は全ペア共通なので 1 回だけ LLM Adapter を通す) ---
    uncond_context = None
    if "uncond" in modes:
        uncond_context = build_llm_adapter_context(dit, prompt_embeds_map[""], device, dtype)

    def context_for(mode: str, pair_idx: int):
        if mode == "zero":
            return None
        if mode == "uncond":
            return uncond_context
        if mode == "caption_shuffled":
            # 1 ローテーションで別ペアのキャプションを割り当てる (ペア順は seed 付き
            # シャッフル済みなのでランダム対応と等価。自分のキャプションには当たらない)
            return build_llm_adapter_context(
                dit, prompt_embeds_map[captions[(pair_idx + 1) % len(captions)]], device, dtype
            )
        return build_llm_adapter_context(dit, prompt_embeds_map[captions[pair_idx]], device, dtype)

    # probe 用のストリーミング積算 (train split): モード x ブロックごとに Gram (D+1)^2 と X^T y
    gram = {m: torch.zeros(num_blocks, d_model + 1, d_model + 1, device=device, dtype=torch.float64) for m in modes}
    xty = {m: torch.zeros(num_blocks, d_model + 1, device=device, dtype=torch.float64) for m in modes}
    # eval split のトークンは CPU に保持
    eval_store = {m: [[] for _ in range(num_blocks)] for m in modes}  # per block: list of (feats fp16 cpu, labels cpu)

    # 保存性/応答性のストリーミング平均 (全ペア)
    pres_sum = {m: torch.zeros(num_blocks, dtype=torch.float64) for m in modes}  # unchanged: matched - shuffled
    resp_sum = {m: torch.zeros(num_blocks, dtype=torch.float64) for m in modes}  # changed:   matched - shuffled
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

        idx_ch = changed.nonzero(as_tuple=True)[0]
        idx_un = unchanged.nonzero(as_tuple=True)[0]
        # probe 用にクラスごと上限までサブサンプル (全モード共通 = 対応のある比較になる)
        sub_ch = idx_ch[torch.randperm(len(idx_ch), device=device)[: args.tokens_per_class]]
        sub_un = idx_un[torch.randperm(len(idx_un), device=device)[: args.tokens_per_class]]
        sub_idx = torch.cat([sub_ch, sub_un])
        sub_lab = torch.cat([
            torch.ones(len(sub_ch), device=device), torch.zeros(len(sub_un), device=device)
        ])

        # 保存性の shuffled baseline 用置換 (unchanged / changed 内で独立に)
        perm_un = idx_un[torch.randperm(len(idx_un), device=device)]
        perm_ch = idx_ch[torch.randperm(len(idx_ch), device=device)]

        for mode in modes:
            ctx = context_for(mode, pair_idx)
            feats_c = all_block_hidden_states(dit, lat_c, args.ref_timestep, context=ctx)
            feats_t = all_block_hidden_states(dit, lat_t, args.ref_timestep, context=ctx)

            for k in range(num_blocks):
                fc, ft = feats_c[k], feats_t[k]  # (S, D) fp32, layer-normed

                # --- probe (条件画像の特徴のみ使用) ---
                xs = fc[sub_idx]  # (n, D)
                xs1 = torch.cat([xs, torch.ones(len(xs), 1, device=device)], dim=1).double()
                if is_eval:
                    eval_store[mode][k].append((xs.half().cpu(), sub_lab.cpu()))
                else:
                    gram[mode][k] += xs1.T @ xs1
                    xty[mode][k] += xs1.T @ sub_lab.double()

                # --- preservation / response (cond vs target, matched - shuffled cosine) ---
                fcn = F.normalize(fc, dim=-1)
                ftn = F.normalize(ft, dim=-1)
                cos_un = (fcn[idx_un] * ftn[idx_un]).sum(-1).mean()
                cos_un_shuf = (fcn[idx_un] * ftn[perm_un]).sum(-1).mean()
                cos_ch = (fcn[idx_ch] * ftn[idx_ch]).sum(-1).mean()
                cos_ch_shuf = (fcn[idx_ch] * ftn[perm_ch]).sum(-1).mean()
                pres_sum[mode][k] += (cos_un - cos_un_shuf).double().cpu()
                resp_sum[mode][k] += (cos_ch - cos_ch_shuf).double().cpu()

            del feats_c, feats_t

        if not is_eval:
            n_train_tokens += len(sub_idx)
        pres_cnt += 1
        logger.info(
            f"pair {pair_idx + 1}/{len(pairs)} ({'eval' if is_eval else 'train'}): "
            f"{os.path.basename(cond_path)} changed={changed.float().mean().item():.2%}"
        )

    # --- 線形 probe を閉形式で解き、eval AUROC を計算 (モードごと) ---
    logger.info(f"solving ridge probes (lambda={args.ridge_lambda}, train tokens={n_train_tokens})...")
    eye = torch.eye(d_model + 1, device=device, dtype=torch.float64) * args.ridge_lambda
    aurocs, pres, resp = {}, {}, {}
    for mode in modes:
        vals = []
        for k in range(num_blocks):
            if not eval_store[mode][k]:
                vals.append(float("nan"))
                continue
            w = torch.linalg.solve(gram[mode][k] + eye, xty[mode][k])  # (D+1,)
            xs = torch.cat([f for f, _ in eval_store[mode][k]]).float().to(device)
            ys = torch.cat([l for _, l in eval_store[mode][k]]).to(device)
            scores = xs @ w[:-1].float() + w[-1].float()
            vals.append(auroc(scores.cpu(), ys.cpu()))
        aurocs[mode] = vals
        pres[mode] = (pres_sum[mode] / pres_cnt).tolist()
        resp[mode] = (resp_sum[mode] / pres_cnt).tolist()

    # --- 報告 ---
    print()
    print(f"pairs={len(pairs)} (eval every {args.eval_every}), image={height}x{width}, "
          f"tokens/image={n_tokens}, ref_timestep={args.ref_timestep}, "
          f"context modes={modes}, mean changed fraction={changed_frac_sum / len(pairs):.2%}")

    def _norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    for mode in modes:
        a_m, p_m, r_m = aurocs[mode], pres[mode], resp[mode]
        print()
        print(f"=== context: {mode} ===")
        print("block | probe AUROC | preservation | response | contrast")
        print("      | (semantic)  | (unchanged)  | (changed)| (pres - resp)")
        print("------+-------------+--------------+----------+---------")
        for k in range(num_blocks):
            contrast = p_m[k] - r_m[k]
            print(f"  {k:3d} |    {a_m[k]:.4f}   |    {p_m[k]:+.4f}   | {r_m[k]:+.4f}  | {contrast:+.4f}")

        # --- 推奨 ---
        valid = [k for k in range(num_blocks) if not np.isnan(a_m[k])]
        a = np.array([a_m[k] for k in valid])
        p = np.array([p_m[k] for k in valid])
        combined = _norm(a) + _norm(p)
        top_single = [valid[i] for i in np.argsort(-combined)[:3]]
        best_sem = valid[int(np.argmax(a))]
        shallow_half = [i for i, k in enumerate(valid) if k < num_blocks // 2]
        best_app = valid[shallow_half[int(np.argmax(p[shallow_half]))]] if shallow_half else valid[int(np.argmax(p))]
        print()
        print(f"suggested single ref_block (normalized AUROC + preservation): {top_single}")
        print(f"suggested dual-block concat: shallow={best_app} (best preservation in the shallow half), "
              f"deep={best_sem} (best probe AUROC)")

    # --- モード間比較 (同一ペア・同一トークンサブサンプルによる対比較) ---
    if len(modes) > 1:
        base = "zero" if "zero" in modes else modes[0]
        others = [m for m in modes if m != base]
        print()
        print(f"=== context mode comparison (delta vs '{base}') ===")
        auroc_cols = " | ".join(f"AUROC[{m}]" for m in modes)
        delta_cols = " | ".join(f"dAUROC[{m}]" for m in others)
        pres_cols = " | ".join(f"pres[{m}]" for m in modes)
        print(f"block | {auroc_cols} | {delta_cols} | {pres_cols}")
        print("------+" + "-" * (len(auroc_cols) + len(delta_cols) + len(pres_cols) + 9))
        for k in range(num_blocks):
            a_str = " | ".join(f"{aurocs[m][k]:^{len(f'AUROC[{m}]')}.4f}" for m in modes)
            d_str = " | ".join(f"{aurocs[m][k] - aurocs[base][k]:^+{len(f'dAUROC[{m}]')}.4f}" for m in others)
            p_str = " | ".join(f"{pres[m][k]:^+{len(f'pres[{m}]')}.4f}" for m in modes)
            print(f"  {k:3d} | {a_str} | {d_str} | {p_str}")
        print()
        if "uncond" in modes and "zero" in modes:
            print("Split: 'uncond' - 'zero' = in-distribution effect (a real but empty context).")
        if "caption" in modes and "caption_shuffled" in modes:
            print("       'caption_shuffled' - 'uncond' = any-plausible-text effect,")
            print("       'caption' - 'caption_shuffled' = image-prompt correspondence effect (mismatch")
            print("       detection itself; if positive, direct evidence for prompt-conditioned reference).")
        elif "caption" in modes and "uncond" in modes:
            print("       'caption' - 'uncond' = text-information effect (prompt-conditioned reference, refs #6.4).")

    print()
    print("Reading guide: AUROC = can a linear probe locate the change region from the cond features")
    print("alone (gate material). preservation = position-matched appearance signal in unchanged")
    print("regions (value-path material). Expect AUROC to rise then degrade with depth, preservation")
    print("to decay; the sweet spot is where both are high. With --context caption, an AUROC gain")
    print("over zero/uncond at mid-deep blocks means the frozen DiT computes the prompt-vs-image")
    print("mismatch into its features (the premise of prompt-conditioned reference).")


if __name__ == "__main__":
    main()
