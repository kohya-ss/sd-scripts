# Reference-Conditioned Editing LoRA for MOSS-Image with `moss_image_train_network_edit.py` (experimental) / `moss_image_train_network_edit.py` による MOSS-Image の参照画像条件付き編集 LoRA（実験的機能）

**This feature is experimental.** The options and the file format may change before (or when) it is merged into the main branch. This page is a short usage note for trying it out; it is not a full guide.

`moss_image_train_network_edit.py` trains a LoRA which generates the target image while looking at a **reference image** (an edit-before / edit-after pair, an inpainting source, etc.). It is a variant of [`moss_image_train_network.py`](moss_image_train_network.md): everything in that guide (model files, caching, sample generation, memory options) applies here, and this page only describes the additions.

<details>
<summary>日本語</summary>

**この機能は実験的です。** オプションやファイル形式は main へのマージ前（またはマージ時）に変わる可能性があります。このページは試用のための短いメモで、網羅的なガイドではありません。

`moss_image_train_network_edit.py` は、**参照画像**（編集前 / 編集後のペア、inpainting の元画像など）を見ながら目標画像を生成する LoRA を学習します。[`moss_image_train_network.py`](moss_image_train_network.md) の派生で、そのガイドの内容（モデルファイル、キャッシュ、サンプル生成、メモリ関連オプション）はそのまま当てはまります。ここでは追加分だけを説明します。

</details>

## 1. How it works (short) / 仕組み（概略）

The reference image is encoded with the VAE to latents and given to the DiT in one of two ways (`--ref_mode`):

* **`kv` (default)**: the clean reference is run through the DiT once per step (empty prompt, timestep `--ref_timestep`, default 0), and the generation attends to the per-block keys / values of the reference (one-way reference with a K/V cache). The reference tokens do not run the FFN in the generation forward, so this is cheaper than `concat`, and at inference the reference K/V are computed once per image and shared by all steps.
* **`concat`**: the reference tokens are concatenated to the image tokens (bidirectional, Kontext style). Simple and known to work, but the inference cost is about 2.5x of the plain generation at 1024².

With `kv`, a **reference gate table** (`--ref_gate`) can be learned together with the LoRA: one gate per DiT block × timestep band that says how much the block attends to the reference. After training, the table can be binarized (top-K blocks per band) and the LoRA fine-tuned with it, so that the closed blocks skip the reference at inference (about 10% faster than plain `kv` with K=20 of 42). The table is also a map of *which blocks use the reference at which timesteps*. The table is stored in the LoRA file (`ref_gate.*` keys; other tools ignore them).

<details>
<summary>日本語</summary>

参照画像は VAE で latent にエンコードされ、`--ref_mode` で選ぶ 2 つの方法のどちらかで DiT に渡されます。

* **`kv`（デフォルト）**: clean な参照を毎ステップ DiT に 1 回通し（空プロンプト、timestep は `--ref_timestep`、デフォルト 0）、生成側が参照の block ごとの key / value に注意を向けます（一方向の KV cache 参照）。参照トークンは生成側の FFN を通らないので `concat` より軽く、推論では参照の K/V を画像ごとに 1 回計算して全ステップで共有します。
* **`concat`**: 参照トークンを画像トークンに連結します（双方向、Kontext 型）。単純で実績のある方式ですが、1024² での推論コストは参照なしの約 2.5 倍です。

`kv` では **参照 gate 表**（`--ref_gate`）を LoRA と一緒に学習できます。DiT の block × timestep 帯ごとに「その block がどれだけ参照を見るか」の gate を持つ表です。学習後に表を 2 値化（帯ごとに上位 K block）して LoRA を再学習すると、閉じた block は推論で参照を見なくなり、`kv` 単体より速くなります（42 block 中 K=20 で約 10%）。表は「どの block がどの timestep で参照を使うか」の地図でもあります。表は LoRA ファイルに同梱されます（`ref_gate.*` キー。他のツールは無視します）。

</details>

## 2. Dataset / データセット

The dataset uses the **ControlNet dataset format**: `image_dir` holds the target images and their captions (the caption describes the *target*), and `conditioning_data_dir` holds the reference images with the **same file names**. `--masked_loss` cannot be used (the conditioning images are references, not masks).

```toml
[general]
enable_bucket = true
bucket_reso_steps = 16
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "data/targets"              # target (edited) images and captions
  conditioning_data_dir = "data/refs"     # reference (source) images, same file names
  num_repeats = 1
  caption_dropout_rate = 0.1
```

Inpainting-like tasks (`--ref_mask_source`): the hole of the reference can come from the alpha channel of RGBA reference images (`alpha`, alpha 255 = hole) or be cut randomly at every step (`random`; set `conditioning_data_dir` to the same directory as `image_dir`, so the reference is the target itself). The hole is applied in the latent space (the clean reference latents are blended with `--ref_mask_fill_latent`, default 0), so the reference latents can still be cached.

<details>
<summary>日本語</summary>

データセットは **ControlNet データセット形式** です。`image_dir` に目標画像とキャプション（キャプションは *目標* の説明）、`conditioning_data_dir` に **同名の** 参照画像を置きます。`--masked_loss` は使えません（conditioning image はマスクではなく参照画像です）。

inpainting 系のタスク（`--ref_mask_source`）では、参照の穴を RGBA 参照画像のアルファ（`alpha`、アルファ 255 = 穴）から取るか、毎ステップランダムに開けます（`random`。`conditioning_data_dir` を `image_dir` と同じにして、参照 = 目標画像そのものにします）。穴は latent 空間で適用される（clean な参照 latent を `--ref_mask_fill_latent`、デフォルト 0 と混合）ので、参照 latent のキャッシュと併用できます。

</details>

## 3. Training / 学習

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 moss_image_train_network_edit.py \
  --pretrained_model_name_or_path "<DiT 00001-of-00003>" --text_encoder "<Qwen3.5-9B>" --vae "<flux2-vae>" \
  --dataset_config edit_dataset.toml --output_dir out --output_name moss_edit --save_model_as safetensors \
  --network_module networks.lora_generic --network_dim 16 --network_train_unet_only \
  --learning_rate 1e-4 --optimizer_type AdamW8bit --lr_scheduler constant \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --mixed_precision bf16 --gradient_checkpointing --attn_mode flash \
  --cache_latents_to_disk --cache_text_encoder_outputs_to_disk \
  --cache_ref_latents_to_disk --ref_latents_cache_dir out/ref_cache \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --ref_dropout_rate 0.05 \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --max_train_epochs 10 --save_every_n_epochs 1
```

* `--cache_ref_latents` / `--cache_ref_latents_to_disk` / `--ref_latents_cache_dir`: cache the VAE latents of the reference images like the target latents (`.npz` next to the reference images, or in `--ref_latents_cache_dir`). Recommended; required with `--replica_devices`.
* `--ref_dropout_rate`: rate to replace the reference with zeros during training, so that the model also works without a reference (reference-free CFG).
* Sample prompts: add `--cn ref.png` to a prompt line to give a reference image (resized to the sample size), and `--mk mask.png` (255 = hole) for inpainting-like LoRA.
* Constraints: `--ref_mode kv` cannot be used with `--blocks_to_swap`. The continuous gate of `--ref_gate` (first stage) cannot be used with `--attn_mode sageattn` (use flash or torch); the binarized second stage can.

### Gate table (optional, `--ref_mode kv` only) / gate 表（任意、`--ref_mode kv` のみ）

Stage 1: learn the table with the LoRA.

```bash
  ... --ref_gate --ref_gate_budget 20 --ref_gate_budget_weight 0.01
```

* `--ref_gate_budget K`: target number of open blocks per band (a hinge penalty; without it the table is not pushed to be sparse). `--ref_gate_lr` (default 1e-2) is the learning rate of the table.
* `--ref_gate_bands`: upper boundaries of the timestep bands (sigma × 1000, descending). Default `950,850,700,500` (5 bands). The number of bands is the number of boundaries + 1; `--ref_gate_bands ""` gives a single band over all timesteps. The bands are trained with the normal timestep distribution (`--timestep_sampling` etc.), so a band that distribution rarely hits is trained slowly; use `--timestep_sampling uniform` or coarser bands if needed.
* `python tools/moss_image_ref_gate_map.py out/moss_edit.safetensors --budget 20` prints the learned table (the map) and the blocks which would stay open per band.

Stage 2: binarize the table (top-K blocks per band, frozen) and fine-tune the LoRA with it. Start from the stage-1 file; the bands are taken from it.

```bash
  ... --ref_gate_binarize --ref_gate_budget 20 --network_weights out/moss_edit.safetensors
```

<details>
<summary>日本語</summary>

* `--cache_ref_latents` / `--cache_ref_latents_to_disk` / `--ref_latents_cache_dir`: 参照画像の VAE latent を目標 latent と同様にキャッシュします（参照画像の隣の `.npz`、または `--ref_latents_cache_dir`）。推奨。`--replica_devices` では必須です。
* `--ref_dropout_rate`: 学習中に参照をゼロに置き換える割合。参照なしでも動く（参照なし CFG）ようにします。
* サンプルプロンプト: 行に `--cn ref.png` を足すと参照画像（サンプルサイズにリサイズ）を与えられます。inpainting 系の LoRA では `--mk mask.png`（255 = 穴）も指定できます。
* 制約: `--ref_mode kv` は `--blocks_to_swap` と併用できません。`--ref_gate` の連続 gate（第 1 段階）は `--attn_mode sageattn` では使えません（flash か torch を使ってください）。2 値化した第 2 段階は使えます。

**gate 表（任意）**

第 1 段階: 表を LoRA と一緒に学習します。

* `--ref_gate_budget K`: 帯ごとの開いた block 数の目標（hinge 罰則。指定しないと表は疎になりません）。`--ref_gate_lr`（デフォルト 1e-2）は表の学習率です。
* `--ref_gate_bands`: timestep 帯の境界（sigma × 1000、降順）。デフォルト `950,850,700,500`（5 帯）。帯の数は境界の数 + 1 で、`--ref_gate_bands ""` なら全 timestep で 1 帯です。各帯は通常の timestep 分布（`--timestep_sampling` 等）で学習されるので、その分布がほとんど当たらない帯は学習が遅くなります。必要なら `--timestep_sampling uniform` や粗い帯割りを使ってください。
* `python tools/moss_image_ref_gate_map.py out/moss_edit.safetensors --budget 20` で学習した表（地図）と、帯ごとに開いたままになる block を表示します。

第 2 段階: 表を 2 値化（帯ごとに上位 K block、凍結）して LoRA を再学習します。第 1 段階のファイルから始めます。帯はそのファイルから取ります。

</details>

## 4. Inference / 推論

ComfyUI has no way to give a reference image to MOSS-Image yet, so use `moss_image_minimal_inference.py` of this repository:

```bash
python moss_image_minimal_inference.py --dit "<DiT>" --text_encoder "<Qwen3.5-9B>" --vae "<flux2-vae>" \
  --lora_weight out/moss_edit.safetensors --ref_image ref.png \
  --prompt "a photo of ..." --image_size 1024 1024 --seed 42 --save_path output
```

* `--ref_image` (or `--ri path` in a prompt line of `--from_file` / interactive mode): the reference image, resized to the image size.
* `--ref_mode`, `--ref_timestep`, `--ref_rope_r`, `--ref_kv_blocks` must match the training (they are recorded in the LoRA metadata as `ss_ref_mode` etc.). The defaults match the training defaults.
* Inpainting-like LoRA: `--ref_mask mask.png` (`--rm` per line, 255 = hole) or `--ref_mask_from_alpha`, and `--ref_mask_fill_latent` as in the training.
* A gate table in the LoRA file is detected automatically and used binarized (gate > 0.5 opens the block; closed blocks skip the reference). `--ref_gate_continuous` uses the continuous gates instead (not with sageattn), `--no_ref_gate` ignores the table.

<details>
<summary>日本語</summary>

ComfyUI には現時点で MOSS-Image に参照画像を渡す仕組みがないので、当リポジトリの `moss_image_minimal_inference.py` を使ってください。

* `--ref_image`（`--from_file` / インタラクティブモードではプロンプト行の `--ri path`）: 参照画像。画像サイズにリサイズされます。
* `--ref_mode`、`--ref_timestep`、`--ref_rope_r`、`--ref_kv_blocks` は学習時と合わせてください（LoRA のメタデータに `ss_ref_mode` 等として記録されています）。デフォルトは学習側のデフォルトと一致します。
* inpainting 系の LoRA: `--ref_mask mask.png`（行単位は `--rm`、255 = 穴）または `--ref_mask_from_alpha`、`--ref_mask_fill_latent` を学習時と同じにします。
* LoRA ファイル内の gate 表は自動検出され、2 値化して使われます（gate > 0.5 の block が開く。閉じた block は参照を見ません）。`--ref_gate_continuous` で連続 gate のまま使い（sageattn 不可）、`--no_ref_gate` で表を無視します。

</details>

## 5. Options / オプション一覧

| Option | Default | Description |
|---|---|---|
| `--ref_mode {kv, concat}` | `kv` | How the reference is given to the DiT (see §1). |
| `--ref_timestep` | `0.0` | Timestep (sigma × 1000) of the reference forward for `kv`. The reference itself is not noised. |
| `--ref_kv_lora {on, on_no_grad}` | `on` | Apply the LoRA in the reference forward of `kv` and train through it (`on`), or apply it detached (`on_no_grad`, cheaper, untested). |
| `--ref_kv_blocks` | all | Comma separated DiT block indices which attend to the reference K/V (`kv`). Exclusive with `--ref_gate`. |
| `--ref_gate` | off | Learn the reference gate table (`kv` only). |
| `--ref_gate_bands` | `950,850,700,500` | Band boundaries of the table; `""` for a single band. |
| `--ref_gate_lr` | `1e-2` | Learning rate of the table. |
| `--ref_gate_budget` / `--ref_gate_budget_weight` | none / `0.1` | Open blocks per band (hinge penalty) and its weight. Also the K of `--ref_gate_binarize`. |
| `--ref_gate_binarize` | off | Second stage: binarize the table of `--network_weights` and fine-tune the LoRA. |
| `--ref_dropout_rate` | `0.0` | Rate to zero the reference during training (reference-free CFG). |
| `--ref_rope_r` | `1.0` | r coordinate of the reference tokens in the 4D RoPE (the base model always uses 0). |
| `--ref_mask_source {none, alpha, random}` | `none` | Hole of the reference for inpainting-like tasks. |
| `--ref_mask_fill_latent` | `0.0` | Fill value of the hole in the normalized latent space. |
| `--cache_ref_latents` / `--cache_ref_latents_to_disk` / `--ref_latents_cache_dir` | off | Cache the reference latents (memory / disk / disk in a directory). |

<details>
<summary>日本語</summary>

| オプション | デフォルト | 説明 |
|---|---|---|
| `--ref_mode {kv, concat}` | `kv` | 参照を DiT に渡す方法（§1）。 |
| `--ref_timestep` | `0.0` | `kv` の参照 forward の timestep（sigma × 1000）。参照自体にノイズは乗せません。 |
| `--ref_kv_lora {on, on_no_grad}` | `on` | `kv` の参照 forward に LoRA を適用し勾配も流す（`on`）か、適用のみ（`on_no_grad`、軽いが未検証）。 |
| `--ref_kv_blocks` | 全 block | 参照 K/V に注意を向ける DiT block 番号（カンマ区切り、`kv`）。`--ref_gate` と排他。 |
| `--ref_gate` | off | 参照 gate 表を学習する（`kv` のみ）。 |
| `--ref_gate_bands` | `950,850,700,500` | 表の帯境界。`""` で 1 帯。 |
| `--ref_gate_lr` | `1e-2` | 表の学習率。 |
| `--ref_gate_budget` / `--ref_gate_budget_weight` | なし / `0.1` | 帯ごとの開く block 数（hinge 罰則）とその重み。`--ref_gate_binarize` の K にも使います。 |
| `--ref_gate_binarize` | off | 第 2 段階: `--network_weights` の表を 2 値化して LoRA を再学習。 |
| `--ref_dropout_rate` | `0.0` | 学習中に参照をゼロにする割合（参照なし CFG）。 |
| `--ref_rope_r` | `1.0` | 4D RoPE での参照トークンの r 座標（ベースモデルは常に 0）。 |
| `--ref_mask_source {none, alpha, random}` | `none` | inpainting 系タスク用の参照の穴。 |
| `--ref_mask_fill_latent` | `0.0` | 正規化 latent 空間での穴の値。 |
| `--cache_ref_latents` / `--cache_ref_latents_to_disk` / `--ref_latents_cache_dir` | off | 参照 latent のキャッシュ（メモリ / ディスク / 指定ディレクトリ）。 |

</details>
