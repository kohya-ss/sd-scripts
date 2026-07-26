# ControlNet-LLLite Training Guide for Anima using `anima_train_control_net_lllite.py` / `anima_train_control_net_lllite.py` を用いた Anima モデルの ControlNet-LLLite 学習ガイド

This document explains how to train a **ControlNet-LLLite** for Anima using `anima_train_control_net_lllite.py`, and how to run minimal inference with the trained weights via `anima_minimal_inference_control_net_lllite.py`.

ControlNet-LLLite is a lightweight, LoRA-like conditional control module originally introduced for SDXL (see [`train_lllite_README.md`](./train_lllite_README.md)). This Anima port retargets it to the DiT (MiniTrainDIT) architecture used by Anima: small adapter modules are attached to selected `Linear` layers of each transformer block, and a shared conditioning-image embedding (`conditioning1`) is broadcast to all of them.

The current implementation is the **v2 architecture**, which extends the original LLLite (SDXL version) with:

* a deeper `conditioning1` trunk (Conv stride-4 ×2 + Conv stride-1 + GroupNorm + SiLU + ResBlocks + final `LayerNorm`) so that the shared conditioning embedding has a wider receptive field,
* **FiLM (γ, β)** modulation inside each LLLite module on top of the original concat-then-mid path, zero-initialized so the module starts from identity,
* a per-module **depth embedding** (zero-init bias) added to the shared `cond_emb`, used as a probe for layer-specificity,
* atomic `target_layers` specifiers, including a new **`mlp_fc1_pre`** target that injects LLLite into the MLP block,
* an optional **ASPP** (Atrous Spatial Pyramid Pooling) tail in `conditioning1`, switchable via `--lllite_use_aspp`.

**v2.1** additionally offers an optional **latent conditioning input** (`--lllite_cond_input latent`): the control image is VAE-encoded and the resulting 16ch latent is fed to `conditioning1` instead of raw pixels, so the pretrained VAE encoder does the heavy feature extraction (see Section 8). The weight format and `lllite.version` are unchanged (`"2"`); pixel-space weights keep working exactly as before.

> **Status:** experimental. Currently supports image generation only (`T=1`). `--blocks_to_swap`, `--cpu_offload_checkpointing`, `--unsloth_offload_checkpointing`, `--deepspeed`, and `--fused_backward_pass` are not yet supported and the training script will assert if any of them is enabled.

An experimental ComfyUI ControlNet-LLLite node for Anima is also available [here](https://github.com/kohya-ss/ComfyUI-Anima-LLLite).

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts` リポジトリに含まれる `anima_train_control_net_lllite.py` を用いて Anima モデル向けの **ControlNet-LLLite** を学習する手順、および学習した重みを `anima_minimal_inference_control_net_lllite.py` で推論する基本的な手順について解説します。

ControlNet-LLLite は SDXL 向けに導入された LoRA ライクな軽量条件付け手法です（オリジナルの解説は [`train_lllite_README-ja.md`](./train_lllite_README-ja.md) を参照）。Anima 版では、Anima が採用する DiT (MiniTrainDIT) アーキテクチャに移植してあり、各 Transformer ブロックの選択した `Linear` レイヤに小さな adapter を貼り、conditioning 画像を埋め込んだ単一の `conditioning1` を全モジュールに配布する構成になっています。

現在の実装は **v2 アーキテクチャ**で、初期実装（SDXL版LLLite）に対して以下を拡張しています：

* `conditioning1` を深層化（Conv stride-4 ×2 + Conv stride-1 + GroupNorm + SiLU + ResBlock + 末尾 `LayerNorm`）し、shared conditioning 埋め込みの受容野を広げています。
* 各 LLLite モジュール内で、従来の concat→mid 経路に加えて **FiLM (γ, β)** による変調を導入。zero-init で identity から学習開始します。
* モジュール毎の **depth embedding**（zero-init bias）を shared `cond_emb` に加算し、層別性の必要度を観察するプローブとして利用しています。
* `target_layers` を atomic specifier 形式でも指定できるよう拡張し、新しく **`mlp_fc1_pre`**（MLP ブロックへの注入）を追加しました。
* `conditioning1` 末尾に **ASPP** (Atrous Spatial Pyramid Pooling) を任意で挿入できる `--lllite_use_aspp` を追加しました。

**v2.1** ではさらに、制御画像を VAE で encode した 16ch latent を（pixel の代わりに）`conditioning1` に入力する **latent 条件入力**（`--lllite_cond_input latent`）を選べるようになりました。学習済み VAE encoder に特徴抽出の大半を担わせる狙いです（第 8 節）。重みフォーマットと `lllite.version`（`"2"`）は据え置きで、pixel 空間の既存重みはこれまで通り動作します。

> **ステータス:** 実験的実装です。現状は画像生成（`T=1`）のみ対応しています。`--blocks_to_swap` / `--cpu_offload_checkpointing` / `--unsloth_offload_checkpointing` / `--deepspeed` / `--fused_backward_pass` には未対応で、指定すると学習スクリプトが assert で停止します。

実験的なComfyUI用のControlNet-LLLiteノードも [こちら](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) で公開しています。

</details>

## 1. How it Differs from the Standard Anima LoRA Script / 通常の Anima LoRA 学習との違い

`anima_train_control_net_lllite.py` is derived from `anima_train.py` but trains **only** the ControlNet-LLLite adapter; the DiT itself is fully frozen.

| | `anima_train_network.py` (LoRA) | `anima_train_control_net_lllite.py` |
|---|---|---|
| Target | DiT LoRA | ControlNet-LLLite adapter only (DiT frozen) |
| Dataset | DreamBooth / fine-tuning | **ControlNet format** (image + conditioning image) |
| Network module | `--network_module=networks.lora_anima` | (none — built-in `ControlNetLLLiteDiT`) |
| Extra inputs at train step | — | `conditioning_images` from each batch |
| Saved weights | LoRA `.safetensors` | LLLite `.safetensors` (`lllite_conditioning1.*` + per-module `{lllite_name}.*`, e.g. `lllite_dit_blocks_0_self_attn_q_proj.*`) |

The dataset format is the same as the existing SDXL ControlNet-LLLite script. See the **Preparing the dataset** section of [`train_lllite_README.md`](./train_lllite_README.md#preparing-the-dataset) ([日本語](./train_lllite_README-ja.md#データセットの準備)) for the directory layout, `conditioning_data_dir`, and dataset synthesis tips.

<details>
<summary>日本語</summary>

`anima_train_control_net_lllite.py` は `anima_train.py` の派生で、DiT 本体は完全に凍結し **ControlNet-LLLite adapter のみ**を学習します。

| | `anima_train_network.py`（LoRA） | `anima_train_control_net_lllite.py` |
|---|---|---|
| 学習対象 | DiT の LoRA | ControlNet-LLLite adapter のみ（DiT は凍結） |
| データセット形式 | DreamBooth / fine-tuning | **ControlNet 形式**（教師画像 + conditioning 画像） |
| Network module | `--network_module=networks.lora_anima` | （不要、`ControlNetLLLiteDiT` 内蔵） |
| 学習ステップの追加入力 | — | バッチ内の `conditioning_images` |
| 保存される重み | LoRA `.safetensors` | LLLite `.safetensors`（`lllite_conditioning1.*` と モジュール毎の `{lllite_name}.*`、例：`lllite_dit_blocks_0_self_attn_q_proj.*`） |

データセット形式は既存の SDXL 向け ControlNet-LLLite と同一です。ディレクトリ構成、`conditioning_data_dir` の指定、データセット合成のヒントなどは [`train_lllite_README-ja.md`](./train_lllite_README-ja.md#データセットの準備) を参照してください。

</details>

## 2. Preparation / 準備

The same model files as ordinary Anima training are required. See [`anima_train_network.md` Section 3](./anima_train_network.md#3-preparation--準備) for details.

In addition you need:

* A **paired dataset** of training images and conditioning images (e.g. lineart, canny, depth) saved with matching basenames. Either a TOML `dataset_config` describing `conditioning_data_dir`, or the CLI form `--train_data_dir <dir> --conditioning_data_dir <dir>` (subset-by-subdir layout) is supported.
* Optionally, a `prompts.txt` with `--cn <path>` (and `--am <float>`) entries for sample-image generation during training.

<details>
<summary>日本語</summary>

通常の Anima 学習で必要なモデルファイル群（DiT、Qwen3、Qwen-Image VAE、LLM Adapter、T5 トークナイザ）が同様に必要です。詳細は [`anima_train_network.md` セクション 3](./anima_train_network.md#3-preparation--準備) を参照してください。

加えて以下が必要です：

* 教師画像と conditioning 画像（例: lineart、canny、depth）を同じベース名でペア化した**データセット**。TOML の `dataset_config` で `conditioning_data_dir` を指定する方法、もしくは CLI で `--train_data_dir <dir> --conditioning_data_dir <dir>`（サブディレクトリ単位の自動 subset 生成）を指定する方法、どちらも使えます。
* 学習中のサンプル画像生成を行いたい場合は、`--cn <path>`（任意で `--am <float>`）を含む `prompts.txt`。

</details>

## 3. Running the Training / 学習の実行

Example command (one line in practice — line breaks shown for readability; use `\` on Linux/macOS or `^` on Windows to wrap):

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_control_net_lllite.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path to Qwen3-0.6B model or directory>" \
  --vae="<path to Qwen-Image VAE model>" \
  --dataset_config="my_anima_lllite_dataset.toml" \
  --output_dir="<output directory>" \
  --output_name="my_anima_lllite" \
  --save_model_as=safetensors \
  --cond_emb_dim=32 \
  --lllite_mlp_dim=64 \
  --lllite_cond_dim=64 \
  --lllite_target_layers=self_attn_q \
  --learning_rate=5e-5 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sampling="shift" \
  --discrete_flow_shift=3.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --vae_chunk_size=64 \
  --vae_disable_cache
```

A minimal dataset TOML for the ControlNet format looks like:

```toml
[general]
caption_extension = ".txt"
shuffle_caption = false

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/path/to/training_images"
  conditioning_data_dir = "/path/to/conditioning_images"
  num_repeats = 1
```

For a fuller description of dataset options, see the SDXL LLLite guide ([English](./train_lllite_README.md#preparing-the-dataset) / [日本語](./train_lllite_README-ja.md#データセットの準備)) and the [Dataset Configuration Guide](./config_README-en.md). The dataset format and the meaning of `conditioning_data_dir` are identical to the SDXL version.

<details>
<summary>日本語</summary>

学習の実行コマンド例は英語側を参照してください（実際は1行で書くか、Linux/macOS では `\`、Windows では `^` で改行してください）。

ControlNet 形式のデータセット TOML の最小例も英語側にある通りで、`conditioning_data_dir` の指定が SDXL LLLite と同一です。データセット設定の詳細については SDXL LLLite ガイド（[`train_lllite_README-ja.md`](./train_lllite_README-ja.md#データセットの準備)）と [データセット設定ガイド](./config_README-ja.md) を参照してください。

</details>

### 3.1. LLLite-Specific Arguments / LLLite 固有の引数

The following options are unique to this script. Anima-related arguments (`--qwen3`, `--vae`, `--llm_adapter_path`, `--t5_tokenizer_path`, `--timestep_sampling`, `--discrete_flow_shift`, etc.) and common training arguments (`--learning_rate`, `--optimizer_type`, `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`, ...) behave the same as in [`anima_train_network.md`](./anima_train_network.md).

* `--cond_emb_dim=<int>` (default `32`)
  * Channel dimension of the conditioning-image embedding produced by `conditioning1`. This is the width of the shared `cond_emb` distributed to every LLLite module. Larger values give more capacity to the conditioning representation.

* `--lllite_cond_dim=<int>` (default `64`)
  * Internal trunk channel width of `conditioning1`. The two stride-4 convs progress as `3 → cond_dim/2 → cond_dim`, ResBlocks (and optional ASPP) operate at this width, and a final 1×1 conv squeezes back to `cond_emb_dim`. Acts as a separate capacity knob from `--cond_emb_dim`.

* `--lllite_cond_resblocks=<int>` (default `1`)
  * Number of pre-activation `(GN→SiLU→Conv3×3→GN→SiLU→Conv3×3 + skip)` ResBlocks inserted in `conditioning1` after the stride-16 stack. Increasing this widens the receptive field of the shared conditioning embedding. `0` disables ResBlocks entirely. Since `conditioning1` is computed once per step (and once per inference call), making it deeper costs essentially nothing at inference time.

* `--lllite_use_aspp` (flag, default off)
  * Append an ASPP (Atrous Spatial Pyramid Pooling) tail to `conditioning1` (after the ResBlocks, before the final 1×1 projection). The ASPP has parallel branches for dilations `(1, 2, 4, 8)` plus a global-pool branch, concatenated and projected back to `cond_dim`. This explicitly injects multi-scale and global summary information into the shared conditioning, which is more useful for cond signals that depend on global structure (e.g. depth, segmentation) than for purely local ones (e.g. lineart).

* `--lllite_mlp_dim=<int>` (default `64`)
  * Hidden dimension of the LoRA-like down / mid / up Linear path inside each LLLite module. Also drives the FiLM `(γ, β)` projection (`cond_emb_dim → 2 × mlp_dim`). Analogous to `network_dim` in standard LoRA.

* `--lllite_target_layers=<spec>` (default `self_attn_q`)
  * Selects which `Linear` layers receive an LLLite module. Accepts either a **preset** name or a comma-separated list of **atomic specifiers**.
  * Atomic specifiers:
    * `self_attn_q_pre` — `self_attn.q_proj` (input perturbation).
    * `self_attn_kv_pre` — `self_attn.k_proj` and `self_attn.v_proj` together.
    * `cross_attn_q_pre` — `cross_attn.q_proj`.
    * `mlp_fc1_pre` — `mlp.layer1` (the fc1 of the GPT2-style feed-forward).
  * Presets (back-compat aliases for common combinations):
    * `self_attn_q` ≡ `self_attn_q_pre` — lightest (~1 module per block).
    * `self_attn_qkv` ≡ `self_attn_q_pre,self_attn_kv_pre` — adds K/V.
    * `self_attn_qkv_cross_q` ≡ `self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre`.
  * Atomic specifiers can be freely combined: e.g. `self_attn_q_pre,mlp_fc1_pre` injects into Q + MLP fc1, `self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre` covers self-attn QKV + MLP. The resolved canonical atomic set is also written to the weights metadata (see Section 5).
  * `cross_attn.{k,v}_proj` and any `output_proj` are always skipped (the former is incompatible with the conditioning sequence shape; the latter is reserved for future post-projection variants).

* `--lllite_cond_input={pixel,latent}` (default `pixel`)
  * Input space of the control image. `pixel` (default, v2) feeds the image directly to the stride-16 `conditioning1` stem. `latent` (v2.1) VAE-encodes the control image and feeds the 16ch latent to a dedicated latent stem instead. See Section 8 for details and trade-offs.

* `--lllite_dropout=<float>` (default `None`)
  * Dropout applied to the LLLite mid output (post-FiLM, post-SiLU) during training.

* `--lllite_multiplier=<float>` (default `1.0`)
  * Multiplier applied to the LLLite output during training. This same value is used for sample image generation unless overridden per-prompt with `--am`. **Setting this to `0.0` would disable LLLite at training time as well**, so do not use `0` as the global default — use the per-prompt `--am 0` only for inspection (see Section 4).

* `--network_weights=<path>`
  * Path to a pre-trained LLLite `.safetensors` file to resume from. The file is loaded with `strict=False`. The script does not currently enforce that `--lllite_target_layers` matches the metadata of the loaded file, so make sure they agree.

* `--conditioning_data_dir=<dir>`
  * Used only when **not** specifying `--dataset_config`. Together with `--train_data_dir` it produces a single subset-by-subdir dataset.

<details>
<summary>日本語</summary>

本スクリプト固有の引数は以下の通りです。Anima 関連の引数（`--qwen3`、`--vae`、`--llm_adapter_path`、`--t5_tokenizer_path`、`--timestep_sampling`、`--discrete_flow_shift` など）や共通の学習引数（`--learning_rate`、`--optimizer_type`、`--gradient_checkpointing`、`--cache_latents`、`--cache_text_encoder_outputs` ...）の挙動は [`anima_train_network.md`](./anima_train_network.md) と同一です。

* `--cond_emb_dim=<int>`（デフォルト `32`）— `conditioning1` が出力する shared conditioning 埋め込みのチャンネル数（全 LLLite モジュールに配布される `cond_emb` の幅）。大きくすると表現力が増します。
* `--lllite_cond_dim=<int>`（デフォルト `64`）— `conditioning1` 内部 trunk のチャンネル幅。stride-4 の Conv が `3 → cond_dim/2 → cond_dim` と進み、ResBlock や ASPP もこの幅で動作したのち、最後の 1×1 Conv で `cond_emb_dim` に絞られます。`--cond_emb_dim` とは独立に conditioning 内部の表現力を調整する旋盤です。
* `--lllite_cond_resblocks=<int>`（デフォルト `1`）— stride 16 到達後に挿入する pre-activation ResBlock `(GN→SiLU→Conv3×3→GN→SiLU→Conv3×3 + skip)` の段数。増やすと shared conditioning 埋め込みの受容野が広がります。`0` で ResBlock を無効化。`conditioning1` は学習中も推論時も「step あたり 1 回」しか計算されないため、深層化のコストはほぼ無視できます。
* `--lllite_use_aspp`（フラグ、デフォルト OFF）— `conditioning1` の末尾（ResBlocks の後、最終 1×1 Conv の前）に ASPP（Atrous Spatial Pyramid Pooling）を挿入します。dilations は `(1, 2, 4, 8)` の並列ブランチ + global pool ブランチで構成され、concat 後に 1×1 Conv で `cond_dim` に戻します。多スケール受容野と大域要約を明示的に注入する仕組みで、線画のような局所情報主体の cond よりも、depth / segmentation のような大域構造を要する cond で有用な傾向があります。
* `--lllite_mlp_dim=<int>`（デフォルト `64`）— 各 LLLite モジュール内の down / mid / up Linear 路の中間次元。FiLM の `(γ, β)` 出力（`cond_emb_dim → 2 × mlp_dim`）にも影響します。標準 LoRA の `network_dim` 相当です。
* `--lllite_target_layers=<spec>`（デフォルト `self_attn_q`）— LLLite モジュールを貼る `Linear` レイヤを選択します。**preset 名**または**カンマ区切りの atomic specifier** のいずれかで指定できます。
  * atomic specifier:
    * `self_attn_q_pre` — `self_attn.q_proj`（入力摂動）。
    * `self_attn_kv_pre` — `self_attn.k_proj` と `self_attn.v_proj` のセット。
    * `cross_attn_q_pre` — `cross_attn.q_proj`。
    * `mlp_fc1_pre` — `mlp.layer1`（GPT2 系 feed-forward の fc1）。
  * preset（よく使う組合せの後方互換 alias）:
    * `self_attn_q` ≡ `self_attn_q_pre` — 最軽量（1 ブロックあたり 1 モジュール）。
    * `self_attn_qkv` ≡ `self_attn_q_pre,self_attn_kv_pre` — K/V を追加。
    * `self_attn_qkv_cross_q` ≡ `self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre`。
  * atomic specifier は自由に組合せ可能です。例: `self_attn_q_pre,mlp_fc1_pre` は Q + MLP fc1 への注入、`self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre` は self-attn QKV + MLP 。解決後の canonical atomic 列は重みのメタデータにも記録されます（第 5 節参照）。
  * `cross_attn.{k,v}_proj` と `output_proj` は常時スキップされます（前者は context 側との shape 不整合、後者は将来の post-projection 系拡張のために予約）。
* `--lllite_cond_input={pixel,latent}`（デフォルト `pixel`）— 制御画像の入力空間。`pixel`（既定、v2）は画像をそのまま stride-16 の `conditioning1` stem に入力します。`latent`（v2.1）は制御画像を VAE で encode し、16ch latent を専用の latent stem に入力します。詳細と長短は第 8 節を参照してください。
* `--lllite_dropout=<float>`（デフォルト `None`）— LLLite の mid 出力（FiLM 適用後 SiLU 後）に対する学習時 dropout。
* `--lllite_multiplier=<float>`（デフォルト `1.0`）— 学習中の LLLite 出力倍率。サンプル画像生成時にも、prompt 行で `--am` による上書きが無ければこの値が使われます。**`0.0` を指定すると学習時にも LLLite が完全 bypass され grad が乗らず学習が壊れる**ため、グローバルなデフォルトには `0` を使わないでください（観察用途であれば prompt 行で `--am 0` を指定する方法を推奨。第 4 節参照）。
* `--network_weights=<path>` — 続きから学習する場合の LLLite 重み（`.safetensors`）のパス。`strict=False` でロードします。`--lllite_target_layers` と保存時の値が一致しているか確認してください（現状自動チェックはしていません）。
* `--conditioning_data_dir=<dir>` — `--dataset_config` を使わない場合のみ用います。`--train_data_dir` と組み合わせ、サブディレクトリ単位の subset を自動生成します。

</details>

### 3.2. Recommended Starting Settings / 推奨される開始設定

A reasonable starting point for lineart-style control on Anima is the v2 default configuration:

* `--cond_emb_dim=32`
* `--lllite_cond_dim=64`
* `--lllite_cond_resblocks=1`
* `--lllite_mlp_dim=64`
* `--lllite_target_layers=self_attn_q`
* `--learning_rate=1e-4` (roughly half of the SDXL LLLite default; AdaLN-conditioned DiTs tend to be more sensitive to additive bias)
* `--optimizer_type=AdamW8bit`
* `--mixed_precision=bf16`
* `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs`
* ~2,000 image / conditioning pairs at 1024² for ~10 epochs

The v2 baseline already converges faster than the earlier prototype on lineart — clear line-following behavior typically appears within the first epoch.

Tuning hints for harder cond signals:

* For richer or noisier cond (e.g. depth, segmentation), the highest-leverage knobs are usually on the shared `conditioning1` side, since it is computed once and shared by every module: try increasing `--lllite_cond_resblocks` (e.g. `2`–`4`), and/or enabling `--lllite_use_aspp` to inject multi-scale + global summary information.
* If those saturate, expand the per-module coverage by switching `--lllite_target_layers` from `self_attn_q` to `self_attn_qkv`, or by combining MLP injection with `self_attn_q_pre,mlp_fc1_pre` (or `self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre`).
* `--lllite_mlp_dim` and `--cond_emb_dim` are also viable, but they grow the per-module parameter count (and `--lllite_mlp_dim` doubles the FiLM projection width), so prefer the two bullets above first.

Adjusting `--discrete_flow_shift` can also be effective depending on the type of cond signal. For example, for cond with fine details like lineart, trying a smaller value like `--discrete_flow_shift=1.0` may help the learning of fine structures to stabilize by preventing timestep sampling from being too heavily skewed towards the early steps.

These are just example starting points since the optimal hyperparameters may vary based on the dataset, compute budget, and cond signal type. Feedback and experimental results from the community are very welcome.

<details>
<summary>日本語</summary>

線画系の制御を Anima で学習する場合、v2 のデフォルト構成が出発点として実用的です：

* `--cond_emb_dim=32`
* `--lllite_cond_dim=64`
* `--lllite_cond_resblocks=1`
* `--lllite_mlp_dim=64`
* `--lllite_target_layers=self_attn_q`
* `--learning_rate=1e-4`（SDXL LLLite のデフォルトのおよそ半分。AdaLN ベースの DiT は加算成分に対する感度が高めのため）
* `--optimizer_type=AdamW8bit`
* `--mixed_precision=bf16`
* `--gradient_checkpointing`、`--cache_latents`、`--cache_text_encoder_outputs`
* 1024² の画像/条件画像ペアを約 2,000 組、約 10 epoch

v2 baseline は初期実装と比べて線画では収束が速く、典型的には 1 epoch 目から線画追従が明確に見え始めます。

より難しい cond 信号に対する調整指針：

* depth / segmentation のような情報量の多い・大域的な構造を伴う cond では、まず全モジュールが共有する `conditioning1` 側の強化が効率的です。`--lllite_cond_resblocks` を `2`〜`4` に増やす、または `--lllite_use_aspp` を有効にして多スケール・大域要約を注入することを試してください。
* 上記で頭打ちなら、各モジュール側のカバレッジを広げます。`--lllite_target_layers` を `self_attn_q` から `self_attn_qkv` へ、または atomic 指定で `self_attn_q_pre,mlp_fc1_pre`（あるいは `self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre`）に切り替え、MLP への注入も併用します。
* `--lllite_mlp_dim` や `--cond_emb_dim` も選択肢ですが、モジュール毎のパラメータ数が増える（`--lllite_mlp_dim` は FiLM 投影の幅も倍化させる）ため、まずは上の 2 つを試すのが効率的です。

cond 信号の種類に応じて、`--discrete_flow_shift` を調整するのも有効な場合があります。たとえば lineart のような微細な構造を伴う cond では、`--discrete_flow_shift=1.0` のように小さめの値を試すと、timestep sampling が前半に偏りすぎず、微細な構造の学習が安定する可能性があります。
</details>

これらはあくまで出発点の例で、最適なハイパーパラメータはデータセットや計算リソース、cond 信号の種類によって異なる可能性があります。コミュニティからのフィードバックや実験結果を歓迎します。

## 4. Sample Image Generation During Training / 学習中のサンプル画像生成

`--sample_prompts`, `--sample_every_n_epochs`, `--sample_every_n_steps`, `--sample_at_first` work the same as the LoRA training script. To pass a per-prompt control image and (optionally) a per-prompt LLLite multiplier, use the following extras in each `prompts.txt` line:

* `--cn <path>` — control image to feed into LLLite for this prompt.
* `--am <float>` — LLLite multiplier override for this prompt (`additional_network_multiplier`). The first value is used.

Example `prompts.txt`:

```
a cat sitting on a chair --w 1024 --h 1024 --cn lineart_a.png --am 0.8 --d 42
a dog --w 1024 --h 1024 --cn lineart_b.png
inspect base model output --w 1024 --h 1024 --cn lineart_c.png --am 0
```

If `--cn` is omitted (or the file does not exist), the prompt is rendered with the base DiT (LLLite cond cleared) and a warning is logged. The pre-prompt multiplier is saved before each sample and restored afterwards, so an `--am 0` line will not bleed into the next training step.

<details>
<summary>日本語</summary>

`--sample_prompts`、`--sample_every_n_epochs`、`--sample_every_n_steps`、`--sample_at_first` は LoRA 学習スクリプトと同様に動作します。プロンプト毎に control 画像（および LLLite multiplier）を指定するため、`prompts.txt` の各行で以下の追加オプションを使えます：

* `--cn <path>` — このプロンプトで LLLite に与える control 画像。
* `--am <float>` — このプロンプトでの LLLite multiplier 上書き値（`additional_network_multiplier`）。リスト形式で先頭値が使われます。

`--cn` が指定されない、もしくはファイルが存在しない場合は、LLLite の cond を解除した上で素の DiT で生成し、warning を出します。各プロンプトの直前に現在の multiplier を退避し、終了時に復元する仕組みが入っているため、`--am 0` を指定したプロンプトの影響が直後の学習ステップに漏れることはありません。

</details>

## 5. Saved Weights / 保存される重み

The saved `.safetensors` contains only the LLLite-side parameters. On disk the per-module keys use a **named-key format**: each module's tensors are prefixed with its `lllite_name` (e.g. `lllite_dit_blocks_0_self_attn_q_proj`), so a single weight file uniquely identifies which DiT block / Linear each module targets. The internal `state_dict` keys (`lllite_modules.{i}.*` and the stacked `depth_embeds`) are rewritten at save time, and any file that still contains `lllite_modules.*` keys is rejected by `load_lllite_weights` as a legacy format. The on-disk layout is as follows (`{j}` indexes the ResBlock, and `{name}` stands for an `lllite_name` such as `lllite_dit_blocks_0_self_attn_q_proj`):

```
# conditioning1 stem, pixel mode (Conv stride-4 ×2 with an intermediate Conv stride-1)
lllite_conditioning1.conv1.{weight,bias},  lllite_conditioning1.norm1.{weight,bias}
lllite_conditioning1.conv2.{weight,bias},  lllite_conditioning1.norm2.{weight,bias}
lllite_conditioning1.conv3.{weight,bias},  lllite_conditioning1.norm3.{weight,bias}

# conditioning1 stem, latent mode (--lllite_cond_input latent) — replaces conv1..3 above
lllite_conditioning1.lat_conv1.{weight,bias}, lllite_conditioning1.lat_norm1.{weight,bias}
lllite_conditioning1.lat_conv2.{weight,bias}, lllite_conditioning1.lat_norm2.{weight,bias}
# (latent mode + inpainting only) mask conv pyramid 1 -> 4 -> 8 -> 16ch
lllite_conditioning1.mask_conv1.{weight,bias}
lllite_conditioning1.mask_conv2.{weight,bias}
lllite_conditioning1.mask_conv3.{weight,bias}

# (optional) ResBlocks at trunk width
lllite_conditioning1.resblocks.{j}.norm1.{weight,bias}, lllite_conditioning1.resblocks.{j}.conv1.{weight,bias}
lllite_conditioning1.resblocks.{j}.norm2.{weight,bias}, lllite_conditioning1.resblocks.{j}.conv2.{weight,bias}

# (optional, if --lllite_use_aspp) ASPP tail
lllite_conditioning1.aspp.branches.{k}.0.{weight,bias}, lllite_conditioning1.aspp.branches.{k}.1.{weight,bias}
lllite_conditioning1.aspp.global_conv.0.{weight,bias},  lllite_conditioning1.aspp.global_conv.1.{weight,bias}
lllite_conditioning1.aspp.proj.0.{weight,bias},         lllite_conditioning1.aspp.proj.1.{weight,bias}

# final 1x1 projection to cond_emb_dim, then LayerNorm on (B, S, cond_emb_dim)
lllite_conditioning1.proj.{weight,bias}
lllite_conditioning1.out_norm.{weight,bias}

# per LLLite module — keys are prefixed with the module's lllite_name
{name}.depth_embed                  # zero-init, shape (cond_emb_dim,) — split from internal depth_embeds (N, D)
{name}.down.{weight,bias}           # Linear(in_dim -> mlp_dim)
{name}.mid.{weight,bias}            # Linear(mlp_dim + cond_emb_dim -> mlp_dim)
{name}.cond_to_film.{weight,bias}   # Linear(cond_emb_dim -> 2*mlp_dim), zero-init (γ, β)
{name}.up.{weight,bias}             # Linear(mlp_dim -> in_dim), zero-init
```

The metadata records `modelspec.architecture = "anima-preview/control-net-lllite"` plus the following v2-specific keys:

| key | meaning |
|---|---|
| `lllite.version` | architecture version (`"2"`) |
| `lllite.cond_emb_dim` | shared embedding dim |
| `lllite.cond_dim` | `conditioning1` trunk width |
| `lllite.cond_resblocks` | number of ResBlocks |
| `lllite.use_aspp` | `"true"` / `"false"` |
| `lllite.aspp_dilations` | comma-separated dilations (only present when ASPP is on) |
| `lllite.mlp_dim` | per-module MLP / FiLM hidden dim |
| `lllite.target_layers` | the user-supplied `--lllite_target_layers` string verbatim |
| `lllite.target_atomics` | resolved canonical atomic specifier list (comma-separated) |
| `lllite.cond_input_space` | `"pixel"` (v2 default) / `"latent"` (v2.1). Absent in older files → treated as `"pixel"` |
| `lllite.cond_in_channels` | conditioning1 input channel count in **pixel semantics** (`"3"` standard, `"4"` inpaint) — in latent mode the actual stem input is 16 / 32ch |
| `lllite.inpaint_masked_input` | `"true"` if RGB was masked before concat at train time (inpaint only) |

The inference script (Section 6) reads these back, so you normally do not need to specify the architecture options on the command line again. Because the pixel and latent stems use disjoint key names, loading a pixel-space weight file into a latent-mode model (or vice versa) fails immediately with a `cond input space mismatch` error instead of silently leaving the stem at its random initialization. Currently the inference script **requires** the metadata to reconstruct the architecture: state-dict-only auto-detection (e.g. for hand-edited weight files) is not implemented.

Save cadence options (`--save_every_n_epochs`, `--save_every_n_steps`, `--save_state`, `--save_last_n_epochs`, `--save_last_n_steps`, ...) work the same as in standard training scripts.

<details>
<summary>日本語</summary>

保存される `.safetensors` には LLLite 側のパラメータのみが含まれます。ディスク上のモジュール毎キーは **named-key 形式** で、各モジュールのテンソルにそのモジュールの `lllite_name`（例：`lllite_dit_blocks_0_self_attn_q_proj`）が prefix として付きます。これにより重みファイル単独で「どの DiT block のどの Linear 用か」が一意に判別できます。内部 `state_dict` のキー（`lllite_modules.{i}.*` および stack された `depth_embeds`）は保存時に書き換えられ、`lllite_modules.*` キーを含むファイルは `load_lllite_weights` で legacy フォーマットとして reject されます。ディスク上の構造は以下の通りです（`{j}` は ResBlock の index、`{name}` は `lllite_dit_blocks_0_self_attn_q_proj` のような `lllite_name`）：

```
# conditioning1 stem・pixel モード (stride-4 Conv ×2、その間に stride-1 Conv)
lllite_conditioning1.conv1.{weight,bias},  lllite_conditioning1.norm1.{weight,bias}
lllite_conditioning1.conv2.{weight,bias},  lllite_conditioning1.norm2.{weight,bias}
lllite_conditioning1.conv3.{weight,bias},  lllite_conditioning1.norm3.{weight,bias}

# conditioning1 stem・latent モード (--lllite_cond_input latent) — 上記 conv1..3 の代わりに保存される
lllite_conditioning1.lat_conv1.{weight,bias}, lllite_conditioning1.lat_norm1.{weight,bias}
lllite_conditioning1.lat_conv2.{weight,bias}, lllite_conditioning1.lat_norm2.{weight,bias}
# (latent モード + inpainting のときのみ) mask conv pyramid 1 -> 4 -> 8 -> 16ch
lllite_conditioning1.mask_conv1.{weight,bias}
lllite_conditioning1.mask_conv2.{weight,bias}
lllite_conditioning1.mask_conv3.{weight,bias}

# (オプション) trunk 幅で動作する ResBlock
lllite_conditioning1.resblocks.{j}.norm1.{weight,bias}, lllite_conditioning1.resblocks.{j}.conv1.{weight,bias}
lllite_conditioning1.resblocks.{j}.norm2.{weight,bias}, lllite_conditioning1.resblocks.{j}.conv2.{weight,bias}

# (オプション、--lllite_use_aspp 指定時のみ) ASPP 末尾
lllite_conditioning1.aspp.branches.{k}.0.{weight,bias}, lllite_conditioning1.aspp.branches.{k}.1.{weight,bias}
lllite_conditioning1.aspp.global_conv.0.{weight,bias},  lllite_conditioning1.aspp.global_conv.1.{weight,bias}
lllite_conditioning1.aspp.proj.0.{weight,bias},         lllite_conditioning1.aspp.proj.1.{weight,bias}

# 最終 1×1 Conv で cond_emb_dim に絞り、(B, S, cond_emb_dim) に対する LayerNorm
lllite_conditioning1.proj.{weight,bias}
lllite_conditioning1.out_norm.{weight,bias}

# 各 LLLite モジュール — 各キーにそのモジュールの lllite_name が prefix される
{name}.depth_embed                  # zero-init、shape (cond_emb_dim,) — 内部 depth_embeds (N, D) を per-module に split
{name}.down.{weight,bias}           # Linear(in_dim -> mlp_dim)
{name}.mid.{weight,bias}            # Linear(mlp_dim + cond_emb_dim -> mlp_dim)
{name}.cond_to_film.{weight,bias}   # Linear(cond_emb_dim -> 2*mlp_dim), zero-init (γ, β)
{name}.up.{weight,bias}             # Linear(mlp_dim -> in_dim), zero-init
```

メタデータには `modelspec.architecture = "anima-preview/control-net-lllite"` のほか、以下の v2 固有キーが書き込まれます：

| key | 内容 |
|---|---|
| `lllite.version` | アーキテクチャ世代（`"2"`） |
| `lllite.cond_emb_dim` | shared 埋め込み次元 |
| `lllite.cond_dim` | `conditioning1` trunk 幅 |
| `lllite.cond_resblocks` | ResBlock 段数 |
| `lllite.use_aspp` | `"true"` / `"false"` |
| `lllite.aspp_dilations` | カンマ区切り dilations（ASPP 有効時のみ） |
| `lllite.mlp_dim` | LLLite モジュール毎の MLP / FiLM 中間次元 |
| `lllite.target_layers` | ユーザが指定した `--lllite_target_layers` 文字列をそのまま記録 |
| `lllite.target_atomics` | 解決後の canonical atomic specifier 列（カンマ区切り） |
| `lllite.cond_input_space` | `"pixel"`（v2 既定）/ `"latent"`（v2.1）。旧ファイルでキーが無い場合は `"pixel"` 扱い |
| `lllite.cond_in_channels` | conditioning1 入力チャネル数（**pixel 意味論**で `"3"` 通常、`"4"` inpainting）。latent モードでは stem の実入力は 16 / 32ch |
| `lllite.inpaint_masked_input` | 学習時 RGB を mask 域で 0 化してから concat したか（inpaint 専用、`"true"`/`"false"`） |

これらは推論スクリプト（第 6 節）で自動的に読み出されるため、通常はコマンドラインで再指定する必要はありません。pixel と latent の stem はキー名が分離されているため、pixel 重みを latent モードのモデルに（またはその逆で）読ませようとすると、初期値のまま黙って動くのではなく `cond input space mismatch` エラーで即座に失敗します。現在の推論スクリプトはアーキテクチャ復元に**メタデータが必須**で、state_dict 単独からの自動判定（手編集された重みファイル等）には対応していません。

保存頻度の各オプション（`--save_every_n_epochs`、`--save_every_n_steps`、`--save_state`、`--save_last_n_epochs`、`--save_last_n_steps` など）は通常の学習スクリプトと同様に使えます。

</details>

## 6. Minimal Inference / 最低限の推論

`anima_minimal_inference_control_net_lllite.py` extends `anima_minimal_inference.py` (see its docstring for shared behavior — VAE / TE / DiT loading, `--from_file` batch mode, `--interactive`, `--latent_path` decode mode, prompt-line `--w/--h/--d/--s/--g/--fs/--n` overrides, etc.) and adds LLLite attachment. All standard inference options are inherited.

### 6.1. Single-Prompt Example / 単発プロンプトの例

```bash
python anima_minimal_inference_control_net_lllite.py \
  --dit "<path to Anima DiT>" \
  --vae "<path to Qwen-Image VAE>" \
  --text_encoder "<path to Qwen3-0.6B>" \
  --lllite_weights "out/my_anima_lllite-last.safetensors" \
  --control_image "lineart.png" \
  --prompt "a cat sitting on a chair" \
  --image_size 1024 1024 \
  --infer_steps 50 \
  --guidance_scale 3.5 \
  --save_path "out/"
```

### 6.2. Batch Mode / バッチモード

```bash
python anima_minimal_inference_control_net_lllite.py \
  --dit "<...>" --vae "<...>" --text_encoder "<...>" \
  --lllite_weights "out/my_anima_lllite-last.safetensors" \
  --control_image "default.png" \
  --from_file "infer_prompts.txt" \
  --save_path "out/"
```

`infer_prompts.txt` lines may include the standard prompt-line overrides plus two LLLite-specific ones:

* `--cn <path>` — per-prompt control image (overrides `--control_image`).
* `--am <float>` — per-prompt LLLite multiplier (overrides `--lllite_multiplier`).

Example:

```
a cat sitting on a chair --w 1024 --h 1024 --d 42 --cn lineart_a.png --am 0.8
a dog --w 1024 --h 1024 --d 0 --cn lineart_b.png
```

### 6.3. Inference-Only Arguments / 推論専用の引数

* `--lllite_weights <path>` **[required, unless `--latent_path` is given]** — trained LLLite weights.
* `--control_image <path>` — global control image. Required for single-prompt mode; optional in `--from_file` / `--interactive` mode if every prompt provides `--cn`.
* `--lllite_multiplier <float>` (default `1.0`) — global LLLite multiplier.
* `--lllite_cond_emb_dim`, `--lllite_cond_dim`, `--lllite_cond_resblocks`, `--lllite_mlp_dim`, `--lllite_target_layers`, `--lllite_use_aspp`, `--lllite_cond_input` — manual overrides. Normally unnecessary because the values are read from the weights metadata. `--lllite_use_aspp` takes the literal strings `true` / `false`; `--lllite_cond_input` takes `pixel` / `latent`.

When the weights were trained with `--lllite_cond_input latent`, the inference script loads the VAE a second time (kept on CPU between prompts) to encode the control image, so `--vae` must be given as usual.

CFG inference (cond / uncond passes) is handled by simply broadcasting the same `cond_emb` to both passes, so control is applied symmetrically.

<details>
<summary>日本語</summary>

`anima_minimal_inference_control_net_lllite.py` は `anima_minimal_inference.py` を拡張したスクリプトで、VAE / TE / DiT のロード、`--from_file`（バッチ）、`--interactive`、`--latent_path`（latent からの再デコード）、prompt 行での `--w/--h/--d/--s/--g/--fs/--n` オーバーライドなど、既存の推論機能はそのまま継承します。

* 単発推論のコマンド例、バッチ推論のコマンド例、`infer_prompts.txt` の書式は英語側を参照してください。
* バッチ用の追加 prompt 行オプション：
  * `--cn <path>` — このプロンプトでの control 画像（`--control_image` を上書き）。
  * `--am <float>` — このプロンプトでの LLLite 倍率（`--lllite_multiplier` を上書き）。
* 主要な推論専用引数：
  * `--lllite_weights <path>` **[必須、ただし `--latent_path` 指定時を除く]** — 学習済み LLLite 重み。
  * `--control_image <path>` — グローバル control 画像。単発推論では必須。`--from_file` / `--interactive` で全プロンプトが `--cn` を持つ場合は省略可。
  * `--lllite_multiplier <float>`（デフォルト `1.0`）— グローバル LLLite 倍率。
  * `--lllite_cond_emb_dim` / `--lllite_cond_dim` / `--lllite_cond_resblocks` / `--lllite_mlp_dim` / `--lllite_target_layers` / `--lllite_use_aspp` / `--lllite_cond_input` — 通常はメタデータから自動読み込みされるため指定不要。手動上書きが必要な場合のみ指定します。`--lllite_use_aspp` は `true` / `false`、`--lllite_cond_input` は `pixel` / `latent` の文字列リテラルで指定します。
  * `--lllite_cond_input latent` で学習した重みの場合、control 画像の encode 用に VAE をもう一度ロードします（プロンプト間は CPU に退避）。`--vae` は通常通り指定してください。

CFG 推論（cond / uncond の 2 pass）は両 pass に同じ `cond_emb` を配布する形になっており、control は両側に対称に作用します。

</details>

## 7. Inpainting Support / Inpainting 対応

> **Status:** experimental. Adds an optional 4-channel conditioning path (RGB + 1ch binary mask). When enabled, LLLite is trained to fill in the masked region of an image given the rest of it as conditioning.

When `--lllite_cond_in_channels=4` is passed, the LLLite `conditioning1` trunk accepts a 4-channel input instead of the default 3 channels: `[R, G, B, mask]`. The training script generates a fresh random mask per sample at every step (via `library.mask_generator.random_mask`) and concatenates it with the conditioning RGB image. At inference time you pass the control image and the mask separately (the script concatenates them internally).

The default 3-channel behavior (no mask, generic ControlNet usage) is unchanged. The same code path is intentionally generic so that future 4-channel conditioning (e.g. user-supplied control masks for lighting / region control) can reuse the same trunk; for the initial release only inpainting is wired up.

This section describes the pixel path. Inpainting also works with `--lllite_cond_input latent`, where the mask takes a separate route — see Section 8.1.

### 7.1. Mask Convention / マスク規約

* `white` (1.0) = **inpaint area** — the region the model should fill in / 生成すべき穴.
* `black` (0.0) = **keep** — the region whose pixels are preserved.

The mask is binarized at 0.5 on load (`{0.0, 1.0}`) and then rescaled to `{-1.0, 1.0}` via `mask * 2 - 1` before being concatenated as the 4th channel, so that the mask channel matches the `[-1, 1]` range of the RGB conditioning channels.

### 7.2. Dataset Setup / データセット設定

For inpainting, **the conditioning image is the original training image itself**. The simplest way to wire this up without modifying the dataset code is to point `conditioning_data_dir` at the same directory as `image_dir` (or a symlink / copy of it). The same image is then loaded twice per step (once as the training target, once as the LLLite conditioning), and the random mask is generated on the fly inside the training script.

Minimal TOML:

```toml
[general]
caption_extension = ".txt"
shuffle_caption = false

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/path/to/training_images"
  conditioning_data_dir = "/path/to/training_images"  # same as image_dir for inpainting
  num_repeats = 1
```

The CLI form `--train_data_dir <dir> --conditioning_data_dir <dir>` works analogously (point both at the same directory).

### 7.3. Training-Specific Arguments / 学習引数

* `--lllite_cond_in_channels=<int>` (default `3`)
  * `3` = standard ControlNet-LLLite (RGB-only conditioning, no mask).
  * `4` = inpainting (RGB + 1ch mask). Activates random-mask generation per step.

* `--lllite_inpaint_masked_input` (flag, default off)
  * Only effective when `--lllite_cond_in_channels=4`. If set, the RGB channels of the conditioning image are zeroed in the mask region **before** being concatenated with the mask channel (so the model never sees the ground-truth pixels behind the hole). Without this flag, the model sees the full RGB image plus the mask channel — the model is expected to learn to ignore the masked region from the mask signal. The flag is recorded in the saved metadata as `lllite.inpaint_masked_input`.

Example command (one line in practice; line continuations as in Section 3):

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_control_net_lllite.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path to Qwen3-0.6B model or directory>" \
  --vae="<path to Qwen-Image VAE model>" \
  --dataset_config="my_anima_lllite_inpaint.toml" \
  --output_dir="<output directory>" \
  --output_name="my_anima_lllite_inpaint" \
  --save_model_as=safetensors \
  --cond_emb_dim=32 --lllite_mlp_dim=64 --lllite_cond_dim=64 \
  --lllite_target_layers=self_attn_q \
  --lllite_cond_in_channels=4 \
  --learning_rate=1e-4 --optimizer_type="AdamW8bit" --lr_scheduler="constant" \
  --timestep_sampling="shift" --discrete_flow_shift=3.0 \
  --max_train_epochs=10 --save_every_n_epochs=1 \
  --mixed_precision="bf16" --gradient_checkpointing \
  --cache_latents --cache_text_encoder_outputs \
  --vae_chunk_size=64 --vae_disable_cache
```

### 7.4. Sample Image Generation / 学習中のサンプル生成

The sample prompt line accepts an additional flag in inpainting mode:

* `--mk <path>` — per-prompt mask image (white=inpaint, black=keep).

In inpaint mode (`cond_in_channels=4`), if `--mk` is omitted (or the file is not found) the prompt is rendered with the base DiT (LLLite cond cleared) and a warning is logged. The `--cn <path>` flag is still required (it supplies the RGB conditioning image).

Example `prompts.txt`:

```
a cat in a forest --w 1024 --h 1024 --cn original_a.png --mk mask_a.png --d 42
a sunset over mountains --w 1024 --h 1024 --cn original_b.png --mk mask_b.png
```

### 7.5. Inference / 推論

The inference script auto-detects the 4-channel mode from the saved metadata (`lllite.cond_in_channels`, `lllite.inpaint_masked_input`) and requires a mask alongside the control image:

* `--mask_image <path>` — global mask image (required for single-prompt inference when `cond_in_channels=4`).
* Prompt-line `--mk <path>` — per-prompt mask override in `--from_file` / `--interactive` mode.
* `--lllite_cond_in_channels {3,4}` / `--lllite_inpaint_masked_input {true,false}` — manual metadata overrides (usually unnecessary).

Single-prompt example:

```bash
python anima_minimal_inference_control_net_lllite.py \
  --dit "<path to Anima DiT>" --vae "<path to Qwen-Image VAE>" --text_encoder "<path to Qwen3-0.6B>" \
  --lllite_weights "out/my_anima_lllite_inpaint-last.safetensors" \
  --control_image "original.png" --mask_image "mask.png" \
  --prompt "a cat in a forest" --image_size 1024 1024 \
  --infer_steps 50 --guidance_scale 3.5 --save_path "out/"
```

### 7.6. Limitations / 制限

* The model architecture is otherwise unchanged — the DiT itself is not modified for inpainting. All "where to fill" information must flow through the LLLite conditioning path.
* Only random masks are supported during training in this release; future versions may allow user-provided masks via dataset configuration.
* The 4-channel feature is implementation-orthogonal to ASPP, ResBlock count, target_layers, etc. Combine freely.
* `cond_in_channels` other than `3` and `4` are accepted by the architecture (it is generic), but only `4` is wired up to the random-mask training path. Custom values are intended for future user-supplied channels.

<details>
<summary>日本語</summary>

`--lllite_cond_in_channels=4` を指定すると、`conditioning1` の入力チャネルが 3 → 4 になり、`[R, G, B, mask]` を受け付けるようになります。学習時は `library.mask_generator.random_mask` でステップ毎にランダム mask を生成し、`conditioning_images` (= 元画像) と concat して渡します。3ch のデフォルト動作は変更されません。

**マスク規約**: 白 (1.0) = inpaint 対象 (穴) / 黒 (0.0) = 保持。ロード時に 0.5 で二値化 (`{0.0, 1.0}`) した後、`mask * 2 - 1` で `{-1.0, 1.0}` にスケールしてから 4ch 目として concat します (RGB conditioning チャンネルの `[-1, 1]` レンジに合わせるため)。

**データセット**: inpainting では「conditioning 画像 = 元画像」となるため、`conditioning_data_dir` には `image_dir` と同じパス (もしくは symlink / コピー) を指定してください。データセットコード側は無改修です。

**学習引数**:
- `--lllite_cond_in_channels=4`: inpainting モードを有効化 (ステップ毎ランダム mask 生成 + 4ch 化)。
- `--lllite_inpaint_masked_input`: フラグ。立てると RGB を mask 域で 0 化してから concat します (穴の中の真値は見せない)。OFF だと RGB はそのまま渡し、mask チャネル経由で「どこが穴か」を学習させます。メタデータ `lllite.inpaint_masked_input` に記録されます。

**サンプル画像生成**: prompt 行に `--mk <path>` を追加できます。inpaint モードで `--mk` が指定されない／ファイルが見つからない場合は LLLite cond をクリアし素の DiT で生成し warning を出します。

**推論**: メタデータから自動的に 4ch モードを検出します。`--mask_image <path>` (グローバル) または prompt 行 `--mk <path>` (per-prompt) で mask を指定してください。

**制限**: DiT 本体は無改修なので、「どこを埋めるか」の情報は LLLite cond 経路を通る必要があります。現状ランダム mask のみ対応 (将来的にはユーザー指定 mask に対応予定)。`cond_in_channels` の `3`/`4` 以外も構築可能ですが、本リリースでは `4` のみが random-mask 学習パスに接続されています。

</details>

## 8. Latent Conditioning Input (v2.1) / 条件画像の latent 入力 (v2.1)

> **Status:** experimental. Optional; the default remains the v2 pixel path and existing weights are unaffected.

With `--lllite_cond_input latent`, the control image is **VAE-encoded** before it reaches `conditioning1`, and the resulting normalized 16ch latent (the very representation the DiT's `x_embedder` consumes) is fed to a dedicated latent stem instead of the stride-16 pixel stem:

```
pixel  (default): cond image (B,3,H,W)  -> Conv4x4 s4 -> Conv3x3 s1 -> Conv4x4 s4  -> (B,cond_dim,H/16,W/16)
latent          : cond image (B,3,H,W)  -> VAE encode -> (B,16,H/8,W/8)
                                        -> Conv3x3 s1 -> Conv2x2 s2               -> (B,cond_dim,H/16,W/16)
```

The motivation is leverage: `conditioning1` is computed once per step and shared by every LLLite module, so improving it benefits all of them. In pixel mode that feature extractor is trained from scratch on a few thousand pairs; in latent mode the pretrained VAE encoder (a deep ResBlock network) does the bulk of the work for free, and the stem only has to map an already-rich representation into the trunk. FLUX.1 Canny/Depth (Control LoRA) and SD3.5 ControlNet use VAE-encoded control images the same way, including for near-binary signals like canny.

Practical notes:

* The latent stem reaches the token grid exactly: the latent is `H/8`, the stride-2 conv halves it to `H/16`, which is the DiT token resolution (VAE /8 × patch /2).
* Since the VAE carries most of the feature extraction, `--lllite_cond_resblocks 1` is expected to be enough — this is worth checking against your pixel-mode baseline before spending capacity there.
* The control image is encoded **on the fly every step** (under `no_grad`, outside autocast). Caching cond latents to disk is not implemented yet. In latent mode the VAE therefore stays resident on the GPU even with `--cache_latents`, which costs a little VRAM.
* Sparse control signals (lineart, stick-figure poses) are somewhat out of the VAE's reconstruction domain. A cheap sanity check before a long run is to VAE encode→decode a few of your control images and look at the result.
* This is orthogonal to `--lllite_target_layers`, `--lllite_use_aspp`, ResBlock count and multiplier — combine freely.

### 8.1. Inpainting in Latent Mode / latent モードでの inpainting

A mask cannot be pushed through the VAE, so in latent mode (`--lllite_cond_in_channels 4`) it takes a separate route: it is passed at **pixel resolution** as its own tensor and downsampled to latent resolution by a small learnable conv pyramid inside `conditioning1`, then concatenated with the cond latent:

```
mask (B,1,H,W) in {-1,1}
  -> Conv 2x2 s2 -> SiLU      (B, 4, H/2, W/2)   2x2-scale detail
  -> Conv 3x3 s2 -> SiLU      (B, 8, H/4, W/4)   4x4 scale
  -> Conv 3x3 s2 (linear)     (B,16, H/8, W/8)   8x8 scale
concat with the cond latent -> (B,32,H/8,W/8) -> latent stem
```

The pyramid has no normalization between stages on purpose: GroupNorm removes the per-map mean/scale, which for a mostly-uniform mask would erase how much of the image is masked (in the limit, a fully-masked and a fully-unmasked image would produce the same features). The SiLU nonlinearity is what keeps the three stages from collapsing into a single linear map. Total cost is ~1.5K parameters.

Channel balance is 16 (latent) + 16 (mask), and the effective receptive field of the pyramid (~14×14 px) slightly overlaps neighboring 8×8 cells, so boundaries that straddle two latent pixels stay continuous. Each token still sees the mask of exactly the 16×16 pixel patch it represents, matching the per-token mask visibility of the pixel path — the v2 property of clean, blend-free mask boundaries is the requirement this design is built around, and is worth verifying visually when you compare the two modes.

`--lllite_inpaint_masked_input` keeps its meaning: the RGB is zeroed inside the mask region **in pixel space, before** the VAE encode. Mask convention, dataset setup (`conditioning_data_dir` = `image_dir`), random-mask generation and the `--mk` prompt-line flag are all unchanged from Section 7.

Example:

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_control_net_lllite.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path to Qwen3-0.6B model or directory>" \
  --vae="<path to Qwen-Image VAE model>" \
  --dataset_config="my_anima_lllite.toml" \
  --output_dir="<output directory>" --output_name="my_anima_lllite_latent" \
  --save_model_as=safetensors \
  --lllite_cond_input=latent \
  --cond_emb_dim=32 --lllite_mlp_dim=64 --lllite_cond_dim=64 \
  --lllite_cond_resblocks=1 --lllite_target_layers=self_attn_q \
  --learning_rate=1e-4 --optimizer_type="AdamW8bit" --lr_scheduler="constant" \
  --timestep_sampling="shift" --discrete_flow_shift=3.0 \
  --max_train_epochs=10 --save_every_n_epochs=1 \
  --mixed_precision="bf16" --gradient_checkpointing \
  --cache_latents --cache_text_encoder_outputs \
  --vae_chunk_size=64 --vae_disable_cache
```

### 8.2. Limitations / 制限

* Cond latents are not cached to disk; they are re-encoded every step.
* In latent mode `--lllite_cond_in_channels` must be `3` or `4` (the generic `cond_in_channels` path of Section 7 is pixel-mode only).
* Pixel-mode and latent-mode weights are not interchangeable; the loader rejects a mix-up explicitly (Section 5).
* Whether latent conditioning actually beats the pixel stem for a given control signal is an open question — A/B it against a pixel-mode run with the same data and hyperparameters.

<details>
<summary>日本語</summary>

`--lllite_cond_input latent` を指定すると、制御画像を **VAE で encode** してから `conditioning1` に入力します。入力されるのは DiT の `x_embedder` が消費するのと同じ正規化済み 16ch latent で、stride-16 の pixel stem の代わりに専用の latent stem を通ります。

```
pixel（既定）: cond 画像 (B,3,H,W) -> Conv4x4 s4 -> Conv3x3 s1 -> Conv4x4 s4 -> (B,cond_dim,H/16,W/16)
latent       : cond 画像 (B,3,H,W) -> VAE encode -> (B,16,H/8,W/8)
                                   -> Conv3x3 s1 -> Conv2x2 s2              -> (B,cond_dim,H/16,W/16)
```

狙いはレバレッジです。`conditioning1` は 1 step に 1 回だけ計算され全 LLLite モジュールに共有されるため、ここを強化すると全モジュールに効きます。pixel モードではこの特徴抽出器を数千ペアでスクラッチ学習していますが、latent モードでは学習済み VAE encoder（各スケールに ResBlock を持つ深いネット）が追加学習パラメータなしでその大半を担い、stem は「すでにリッチな表現を trunk に写す」だけで済みます。FLUX.1 Canny/Depth (Control LoRA) や SD3.5 ControlNet も同じ方式で、canny のようなニアバイナリ画像でも成立することは実証済みです。

補足:

* latent (`H/8`) を stride 2 で落とすと `H/16` = DiT の token 解像度（VAE /8 × patch /2）にちょうど一致します。
* 特徴抽出の大半を VAE が担うため `--lllite_cond_resblocks 1` で足りる見込みです。pixel モードのベースラインと比較して確認する価値があります。
* 制御画像は**毎ステップ on-the-fly で encode** します（`no_grad`、autocast の外）。cond latent のディスクキャッシュは未実装です。このため latent モードでは `--cache_latents` 指定時も VAE を GPU に常駐させます（その分 VRAM を消費します）。
* 線画やポーズ棒人間のようなスパースな制御信号は VAE の再構成ドメインからやや外れます。長時間の学習前に、手持ちの制御画像を VAE で encode→decode して目視確認しておくと安価な事前検証になります。
* `--lllite_target_layers` / `--lllite_use_aspp` / ResBlock 段数 / multiplier とは直交します。自由に組み合わせられます。

**latent モードでの inpainting**: mask は VAE に通せないため別経路になります。mask は **pixel 解像度のまま別テンソル**で渡し、`conditioning1` 内部の小さな学習可能 conv pyramid で latent 解像度に落としてから cond latent と concat します。

```
mask (B,1,H,W) {-1,1}
  -> Conv 2x2 s2 -> SiLU      (B, 4, H/2, W/2)   2x2 の微細パターン担当
  -> Conv 3x3 s2 -> SiLU      (B, 8, H/4, W/4)   4x4 スケール担当
  -> Conv 3x3 s2 (linear)     (B,16, H/8, W/8)   8x8 スケール担当
cond latent と concat -> (B,32,H/8,W/8) -> latent stem
```

段間に正規化を置いていないのは意図的です。GroupNorm は特徴マップ全体の平均・スケールを除去するため、ほぼ一様な mask では「どれだけの面積がマスクされているか」の情報が失われます（極端には全面マスクと全面非マスクが同じ特徴になります）。3 段が単一の線形写像に縮退するのを防ぐ役割は SiLU が担っています。総パラメータは約 1.5K です。

チャネル配分は latent 16 + mask 16 で対称、pyramid の実効受容野（約 14×14 px）は隣接 8×8 セルに少しはみ出すため、latent 1 画素をまたぐ境界の連続性も拾えます。1 token が担当する 16×16 pixel patch の mask 情報にアクセスできる点は pixel パスと同じです。v2 の「mask 境界のアーティファクトが皆無」という性質を維持することが本設計の必須要件なので、2 モードを比較する際は境界品質を目視で確認してください。

`--lllite_inpaint_masked_input` の意味は変わりません（RGB の mask 域を **pixel 空間で、VAE encode の前に** 0 化します）。マスク規約、データセット設定（`conditioning_data_dir` = `image_dir`）、ランダム mask 生成、prompt 行の `--mk` はすべて第 7 節のままです。

**制限**: cond latent のディスクキャッシュは未対応（毎ステップ再 encode）。latent モードの `--lllite_cond_in_channels` は `3` / `4` のみ（第 7 節の generic な `cond_in_channels` 拡張は pixel モード専用）。pixel 重みと latent 重みは互換性がなく、取り違えは明示的に reject されます（第 5 節）。制御信号ごとに latent が pixel より本当に有利かは未検証なので、同一データ・同一ハイパラでの A/B 比較を推奨します。

</details>

## 9. Semantic Trunk (v3) / semantic trunk (v3)

> **Status:** experimental. Aimed at **semantic editing tasks** (mask-free clothing change, background swap, occlusion removal) where the model must decide *which region of the control image to reject* — a decision that is semantic and global. The default remains the stem trunk (v2/v2.1) and existing weights are unaffected.

With `--lllite_trunk semantic`, `conditioning1`'s input is no longer the cond image/latent itself but the **frozen DiT's own hidden states** of the cond latent: before the denoising forward, the cond latent is run through DiT blocks `0..max(ref_block)` (with a zero text context, so the features depend on the image only), and the output hidden states `(B, 1, H, W, D_model)` feed a lightweight trunk:

```
cond image -> VAE encode -> frozen DiT blocks 0..max(ref_block) (reference forward, no_grad)
  -> LayerNorm(D_model) [per ref block] -> concat -> Linear K*D_model -> cond_dim
                                                             (shared low-rank down projection)
  -> ResBlock x N / (ASPP)                                   (mid-range context)
  -> Conv1x1 -> cond_emb_dim -> LayerNorm                    -> shared cond_emb (B, S, cond_emb_dim)
```

`--lllite_ref_block` accepts either a single block index or a comma-separated list (e.g. `2,13`) for the **dual/multi concat trunk**: one reference forward runs blocks up to the *maximum* listed index and the hidden states of every listed block are collected along the way, LayerNorm'd per block, and concatenated before the shared down projection. Adding a shallower block to a deeper one therefore costs nothing extra at reference time — the motivation is to combine a *preservation-strong* shallow block (probe: block 2 or 7) with a *semantically separable* deeper block (probe: 13–14).

The rationale: the v2/v2.1 stem is trained from scratch, so it cannot learn *semantics* (e.g. "this token is clothing") from a few hundred pairs. The frozen DiT already computes semantic features — the semantic trunk parasitizes the base model for the *understanding* side the same way LLLite always parasitized it for the *generation* side.

Two per-module mechanisms are added on top of the v2 module (`down`/`mid`/`cond_to_film`/`up`):

* **Multiplicative gate** — `Δx = σ(gate([cond_local, h])) ⊙ up(m)`: a per-token scalar gate that decides *where* to inject the condition, separately from the value path that decides *what* to inject. The gate weight is zero-init and the bias starts **open** (σ≈0.88) so gradients reach the value path from step 0; the gate then learns where to close (e.g. the clothing region). The gate map is directly visualizable as the model's predicted change-region mask (`--save_gate_maps` at inference).
* **t-FiLM** — the DiT's own timestep embedding (captured per forward via a hook on `t_embedding_norm`) is projected by a shared zero-init `t_proj` into the cond space and added to `cond_local`, so the fixed reference features are re-interpreted per denoising step.

The reference forward is text-independent and runs under `no_grad`; only blocks up to `ref_block` execute (default `num_blocks // 2`), so the training-step overhead is roughly half a DiT forward. At inference it runs **once per image** before the denoising loop.

Arguments (training):

| Argument | Default | Description |
|---|---|---|
| `--lllite_trunk` | `stem` | `semantic` enables the v3 trunk. Requires `--lllite_cond_input latent`; inpainting (`cond_in_channels=4`) is not supported yet |
| `--lllite_ref_block` | `num_blocks // 2` | DiT block index(es) whose output hidden states feed the trunk. Comma-separated (e.g. `2,13`) enables the dual/multi concat trunk |
| `--lllite_ref_timestep` | `0.0` | timestep of the reference forward, `[0, 1]` scale (0 = clean) |

With the semantic trunk, `--cond_emb_dim` should be raised (128 recommended; a warning is logged below 64): the 32-dim default was sized for the scratch stem and would bottleneck the semantic features. `--lllite_cond_dim` is the trunk width (also 128 recommended). Example:

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_control_net_lllite.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path>" --vae="<path>" \
  --dataset_config="my_anima_lllite.toml" \
  --output_dir="<out>" --output_name="my_anima_lllite_v3" --save_model_as=safetensors \
  --lllite_cond_input=latent --lllite_trunk=semantic \
  --cond_emb_dim=128 --lllite_cond_dim=128 --lllite_mlp_dim=64 \
  --lllite_cond_resblocks=1 --lllite_target_layers=self_attn_qkv_cross_q \
  --learning_rate=1e-4 --optimizer_type="AdamW8bit" --lr_scheduler="constant" \
  --timestep_sampling="shift" --discrete_flow_shift=3.0 \
  --max_train_epochs=10 --save_every_n_epochs=1 \
  --mixed_precision="bf16" --gradient_checkpointing \
  --cache_latents --cache_text_encoder_outputs
```

Saved weight keys (replacing the stem keys of Section 5): `lllite_conditioning1.{ln_in,proj_in,resblocks.*,aspp.*,proj,out_norm,t_proj}` plus a per-module `{lllite_name}.gate.{weight,bias}`. With the dual/multi trunk, `ln_in` becomes a per-block list (`lllite_conditioning1.ln_in.{k}.*`) and `proj_in` widens to `K*D_model`. Metadata gains `lllite.trunk`, `lllite.ref_block` (comma-separated when dual, e.g. `"2,13"`), `lllite.ref_timestep`, and `lllite.version` is `"3"`. Stem and semantic weights are not interchangeable, and neither are single and dual semantic weights; the loader rejects both mix-ups explicitly.

Inference: the metadata is used automatically; `--lllite_ref_block` / `--lllite_ref_timestep` can override it. `--save_gate_maps <dir>` dumps each module's last-step gate map (token-grid grayscale PNG, 1 = copy, 0 = rewrite) plus their mean — the single most useful diagnostic for whether the model has learned to localize the change region.

Limitations:

* Inpainting (mask) is not supported with the semantic trunk yet.
* Reference hidden states are recomputed every training step (no disk cache yet); the step-time overhead is roughly `(max(ref_block) + 1) / num_blocks` of a DiT forward.
* With `--compile`, the blocks see two graph variants (cond cleared during the reference forward vs. set during the main forward), so expect one extra recompile per bucket resolution.
* The gate acts pointwise per token: position *i* can only read the reference at position *i* (widened to mid-range by ResBlocks/ASPP). Edits that require reading the reference at a *different* location (composition changes) are out of scope for this trunk.

<details>
<summary>日本語</summary>

`--lllite_trunk semantic` を指定すると、`conditioning1` の入力が cond 画像/latent そのものではなく、**凍結 DiT 自身が計算した cond latent の中間 hidden states** になります。デノイズ forward の前に cond latent を DiT の block `0..max(ref_block)` に通し（text context はゼロ = 画像のみに依存する特徴）、その出力 `(B, 1, H, W, D_model)` を軽量トランク（LayerNorm → 低ランク Linear → ResBlock/ASPP → 1x1 Conv → LayerNorm）で共有 cond_emb に写します。

`--lllite_ref_block` は単一 index のほか、カンマ区切り（例 `2,13`）で **dual/multi concat トランク**になります。参照フォワードは指定の最大 index まで 1 回だけ実行し、途中の各指定ブロックの hidden states を収集して per-block LayerNorm → 特徴次元 concat → 共有 down 射影（`K*D_model → cond_dim`）に通します。深いブロックに浅いブロックを足す追加コストは参照フォワード側ではゼロです。狙いは、保存性の強い浅層（probe 実測: block 2 / 7）と意味分離性の高い中深層（block 13–14）の組み合わせです。

狙い: v2/v2.1 の stem はスクラッチ学習なので、数百ペアから「このトークンは衣装」といった**意味**を学ぶことは原理的にできません。凍結 DiT は意味特徴を既に計算しているので、LLLite が生成側でベースモデルに寄生してきたのと同じやり方で、**理解側でも寄生**します。マスクなしの衣装変更・背景交換・オクルージョン除去のような「条件画像のどの領域を棄却するかを意味的・大域的に判断するタスク」向けです。

v2 モジュール構成に加わる 2 つの機構:

* **乗算ゲート** — `Δx = σ(gate([cond_local, h])) ⊙ up(m)`。「何を注入するか（値パス）」と「どこで注入するか（ゲート）」を分解する per-token スカラーゲート。weight は zero-init、bias は**開いた状態**（σ≈0.88）で初期化されるため、学習初期から値パスに勾配が流れ、ゲートは後から「閉じるべき場所」（=衣装領域など）を学びます。ゲートマップは「モデルが予測した変更領域マスク」としてそのまま可視化できます（推論時 `--save_gate_maps`）。
* **t-FiLM** — 本体の timestep embedding を（`t_embedding_norm` への forward hook で毎 forward 取得し）共有の zero-init `t_proj` で cond 空間に射影して cond_local に加算します。固定の参照特徴を「今のステップ向けにどう使うか」をアダプタ側が学べます。

参照フォワードはテキスト非依存で `no_grad` 下で走り、`max(ref_block)`（既定 `num_blocks // 2`）までしか実行しないため、学習ステップの増分はおおよそ DiT forward の半分です。推論ではデノイズループ前に**画像あたり 1 回**だけ実行されます。

semantic trunk では `--cond_emb_dim` を引き上げてください（**128 推奨**、64 未満は警告が出ます）。既定の 32 はスクラッチ stem 向けのサイズで、意味特徴のボトルネックになります。`--lllite_cond_dim`（トランク幅）も 128 推奨です。

保存キーは第 5 節の stem キーの代わりに `lllite_conditioning1.{ln_in,proj_in,resblocks.*,aspp.*,proj,out_norm,t_proj}`、各モジュールに `{lllite_name}.gate.{weight,bias}` が加わります。dual/multi では `ln_in` が per-block のリスト（`lllite_conditioning1.ln_in.{k}.*`）になり、`proj_in` は `K*D_model` 幅になります。メタデータには `lllite.trunk` / `lllite.ref_block`（dual ではカンマ区切り、例 `"2,13"`）/ `lllite.ref_timestep` が入り、`lllite.version` は `"3"` になります。stem 重みと semantic 重みは非互換、semantic の single と dual も非互換で、いずれの取り違えもロード時に明示的に reject されます。

制限: inpainting（mask）は未対応。参照 hidden states は毎ステップ再計算（ディスクキャッシュ未実装）。`--compile` 使用時は参照フォワード（cond 解除状態）と本 forward（cond セット状態)で 2 種類のグラフが焼かれるため、バケット解像度あたり recompile が 1 回増えます。ゲートは per-token（位置 i は参照の位置 i しか読めない。ResBlock/ASPP で中域までは拡張）なので、参照の別の場所を読む必要がある構図変更を伴う編集はこのトランクのスコープ外です。

</details>

## 10. Tips & Limitations / 補足と制限

* **Resolution alignment.** The conditioning encoder uses fixed stride 16, so `cond_image` HW must equal `latent HW × 8` (i.e. the original training image size) (the latent is patchified with patch size=2, so stride is 8*2=16). In latent mode (Section 8) the control image is still supplied at the original image size; the VAE /8 plus the stem's stride-2 conv give the same total stride of 16. The DataLoader for the ControlNet dataset already resizes the conditioning image to match the training image, so in practice you only need to make sure the control image you pass at inference time matches the requested `--image_size`.
* **`T=1` only.** Video-style multi-frame inputs are not supported — the wrapper asserts `T==1` at forward time.
* **Bucket size.** The training script enforces a bucket resolution step of 16 (Qwen-Image VAE /8 × patch /2).
* **Memory.** `--blocks_to_swap`, `--cpu_offload_checkpointing`, `--unsloth_offload_checkpointing` are not yet supported. If VRAM is tight, prefer `--full_bf16`, smaller `--lllite_mlp_dim`, lower `--cond_emb_dim`, and `--gradient_checkpointing`.
* **`torch.compile` (speed-up).** Per-block `torch.compile` of the DiT is supported here as well, via the same `--compile` / `--compile_*` arguments shared with the LoRA script. See [`anima_torch_compile.md`](./anima_torch_compile.md) for details and recommended settings. Note that with aspect-ratio bucketing each distinct bucket resolution triggers a one-time recompile, so the first pass over your buckets is slower before it settles.
* **Save format.** The saved `.safetensors` is **not** compatible with the SDXL LLLite format and **not** loadable by `sdxl_gen_img.py`. Use the dedicated inference script in Section 6.
* **Metadata-required at inference.** Inference relies on the architecture metadata (`lllite.version`, `lllite.cond_dim`, `lllite.cond_resblocks`, `lllite.use_aspp`, `lllite.target_atomics`, ...) saved by the training script to reconstruct the LLLite architecture. State-dict-only auto-detection of those fields is not implemented; if a weight file lacks metadata, you currently need to pass the override flags listed in Section 6.3 explicitly.

<details>
<summary>日本語</summary>

* **解像度の整合性.** `conditioning1` の stride は 16 固定なので、`cond_image` の縦横は `latent HW × 8`（つまり元の学習画像サイズ）に一致している必要があります（latent はモデル内で patch size=2 で patchfy されるため、stride は 8*2=16 となる）。latent モード（第 8 節）でも制御画像は元画像サイズのまま渡します（VAE の /8 と stem の stride-2 conv で合計 stride は同じく 16）。ControlNet 形式のデータローダ側で conditioning 画像は教師画像と同じサイズにリサイズされるため、実用上は推論時に渡す control 画像のサイズを `--image_size` と合わせれば OK です。
* **`T=1` のみ.** 動画的な多フレーム入力はサポートしていません（wrapper の forward 冒頭で assert）。
* **bucket サイズ.** 学習スクリプトは bucket 解像度ステップを 16（Qwen-Image VAE /8 × patch /2）として検証します。
* **メモリ.** `--blocks_to_swap`、`--cpu_offload_checkpointing`、`--unsloth_offload_checkpointing` は未対応です。VRAM が厳しい場合は `--full_bf16`、`--lllite_mlp_dim` を下げる、`--cond_emb_dim` を下げる、`--gradient_checkpointing` を有効にする、などで対応してください。
* **`torch.compile`（高速化）.** LoRA 学習スクリプトと共通の `--compile` / `--compile_*` 引数により、本スクリプトでも DiT のブロック単位 `torch.compile` が利用できます。詳細や推奨設定は [`anima_torch_compile.md`](./anima_torch_compile.md) を参照してください。なお aspect-ratio bucketing 使用時はバケット解像度ごとに初回 1 回の recompile が走るため、全バケットを一巡するまでは遅くなりますが、その後は安定します。
* **保存形式.** 保存される `.safetensors` は SDXL LLLite フォーマットとは**互換性がなく**、`sdxl_gen_img.py` ではロードできません。推論には第 6 節の専用スクリプトを使用してください。
* **推論時のメタデータ依存.** 推論時のアーキテクチャ復元は、学習スクリプトが書き込んだメタデータ（`lllite.version` / `lllite.cond_dim` / `lllite.cond_resblocks` / `lllite.use_aspp` / `lllite.target_atomics` など）に依存します。state_dict 単独からの自動判定は実装されていないため、メタデータの無い重みを使う場合は第 6.3 節の手動上書き引数を明示的に指定してください。

</details>
