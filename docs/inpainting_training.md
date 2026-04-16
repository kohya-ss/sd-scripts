# Inpainting Model Training / インペインティングモデルの学習

Inpainting training allows you to train a model that can selectively regenerate masked regions of an image while preserving the rest. The resulting model can be used directly with inpainting-capable UIs (e.g. AUTOMATIC1111, ComfyUI) or as a base for further fine-tuning.

<details>
<summary>日本語</summary>
インペインティング学習では、画像のマスクされた領域を選択的に再生成し、それ以外の部分を保持するモデルを学習できます。学習済みモデルは、インペインティングに対応した UI（AUTOMATIC1111、ComfyUI など）でそのまま使用したり、さらなるファインチューニングのベースとして利用できます。
</details>

## How it works / 仕組み

During training the UNet receives a 9-channel input instead of the usual 4:

| Channels | Content |
|---|---|
| 0–3 | Noisy latents (normal diffusion input) |
| 4 | Downsampled binary mask (1 = region to regenerate) |
| 5–8 | VAE-encoded masked image (original × (1 − mask)) |

Masks are generated **randomly per training step** using a procedural mask generator that produces cloud-shaped (fractional Brownian motion), polygon, and rectangular/ellipse masks, or random combinations of all three. This gives the model a diverse range of inpainting scenarios without requiring manually prepared mask images.

<details>
<summary>日本語</summary>
学習中、UNet は通常の 4 チャンネルではなく 9 チャンネルの入力を受け取ります。

| チャンネル | 内容 |
|---|---|
| 0–3 | ノイズを加えた潜在変数（通常の拡散モデル入力） |
| 4 | ダウンサンプルされた二値マスク（1 = 再生成する領域） |
| 5–8 | VAE でエンコードされたマスク済み画像（元画像 × (1 − マスク)） |

マスクは**学習ステップごとにランダムに生成**されます。プロシージャルマスクジェネレーターが、クラウド状（フラクタルブラウン運動）、ポリゴン、矩形・楕円のマスク、またはそれらの組み合わせを生成します。これにより、手動でマスク画像を用意することなく、多様なインペインティングシナリオをモデルに学習させることができます。
</details>

## Requirements / 必要条件

- An inpainting base model (e.g. `sd-v1-5-inpainting.ckpt`, `sd_xl_base_1.0_inpainting.safetensors`) for best results, **or** a standard model checkpoint. When a standard checkpoint is used, the UNet `conv_in` layer is automatically expanded from 4 to 9 channels (original weights preserved, extra channels zero-initialised) so training can proceed from scratch.
- `--cache_latents` and `--cache_latents_to_disk` **cannot** be used with `--train_inpainting`. Masks are generated randomly each step from the source image, so the source image must be available at training time.

<details>
<summary>日本語</summary>

- 最良の結果を得るにはインペインティング用ベースモデル（例: `sd-v1-5-inpainting.ckpt`、`sd_xl_base_1.0_inpainting.safetensors`）、**または**通常のモデルチェックポイント。通常のチェックポイントを使用した場合、UNet の `conv_in` レイヤーは自動的に 4 チャンネルから 9 チャンネルに拡張されます（元の重みを保持し、追加チャンネルはゼロ初期化）。
- `--train_inpainting` と `--cache_latents` / `--cache_latents_to_disk` は**同時に使用できません**。マスクは各ステップでソース画像からランダムに生成されるため、学習時に元の画像が必要です。
</details>

## Training / 学習

Add `--train_inpainting` to your training command. All training scripts support this flag: `train_network.py`, `sdxl_train_network.py`, `train_db.py`, `sdxl_train.py`, `fine_tune.py`, and `train_textual_inversion.py`.

<details>
<summary>日本語</summary>
学習コマンドに `--train_inpainting` を追加します。このフラグはすべての学習スクリプト（`train_network.py`、`sdxl_train_network.py`、`train_db.py`、`sdxl_train.py`、`fine_tune.py`、`train_textual_inversion.py`）で使用できます。
</details>

### SD1.5 example / SD1.5 の例

```bash
accelerate launch train_network.py \
  --pretrained_model_name_or_path="sd-v1-5-inpainting.ckpt" \
  --dataset_config="my_dataset.toml" \
  --output_dir="./output" \
  --output_name="my_inpainting_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --network_dim=32 \
  --train_inpainting \
  --mixed_precision=bf16 \
  --xformers
```

### SDXL example / SDXL の例

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="sd_xl_base_1.0_inpainting.safetensors" \
  --dataset_config="my_dataset.toml" \
  --output_dir="./output" \
  --output_name="my_sdxl_inpainting_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --network_dim=32 \
  --train_inpainting \
  --mixed_precision=bf16 \
  --optimizer_type=Adafactor \
  --gradient_checkpointing
```

### Dataset configuration / データセット設定

Use a standard DreamBooth or fine-tuning dataset. No special mask images are required — masks are generated automatically during training.

Note: `cache_latents` must be `false` (the default) when using `train_inpainting`.

<details>
<summary>日本語</summary>
標準的な DreamBooth またはファインチューニング用データセットを使用します。専用のマスク画像は不要です — マスクは学習中に自動生成されます。

注意: `train_inpainting` を使用する場合、`cache_latents` は `false`（デフォルト）のままにしてください。
</details>

## Minimal inference script / 最低限の推論スクリプト

`inpainting_minimal_inference.py` provides a self-contained inference script for testing inpainting models without the full training pipeline. It supports both SD1.5 and SDXL inpainting checkpoints and accepts either a user-supplied mask image or a random procedural mask.

```bash
# SD1.5 inpainting with a mask image
python inpainting_minimal_inference.py \
    --ckpt_path sd-v1-5-inpainting.ckpt \
    --image input.png \
    --mask mask.png \
    --prompt "a yawning cat"

# SDXL inpainting with a random procedural (wobbly ellipse) mask
python inpainting_minimal_inference.py \
    --ckpt_path sd_xl_base_1.0_inpainting.safetensors \
    --sdxl \
    --image input.png \
    --prompt "a yawning cat" \
    --width 1024 --height 1024 \
    --seed 42
```

The mask image should be the same size as the source image, with white pixels indicating the region to regenerate and black pixels indicating the region to preserve. If `--mask` is omitted, a random procedural mask is generated using `library/mask_generator.py`.

<details>
<summary>日本語</summary>
`inpainting_minimal_inference.py` は、学習パイプライン全体を必要とせずにインペインティングモデルをテストするための、独立した推論スクリプトです。SD1.5 および SDXL のインペインティングチェックポイントに対応しており、ユーザー指定のマスク画像またはランダムなプロシージャルマスクを使用できます。

マスク画像はソース画像と同じサイズで、白いピクセルが再生成する領域、黒いピクセルが保持する領域を示します。`--mask` を省略すると、`library/mask_generator.py` を使ってランダムなプロシージャルマスクが生成されます。
</details>

## Loading existing inpainting checkpoints / 既存インペインティングチェックポイントの読み込み

The training scripts automatically detect whether a checkpoint has a 9-channel `conv_in` weight and configure the UNet accordingly. You do not need to pass any extra flags to load an inpainting checkpoint — it is handled transparently.

<details>
<summary>日本語</summary>
学習スクリプトは、チェックポイントの `conv_in` の重みが 9 チャンネルかどうかを自動的に検出し、UNet を適切に設定します。インペインティングチェックポイントを読み込むために追加のフラグを指定する必要はありません — 透過的に処理されます。
</details>

## Sample images during training / 学習中のサンプル画像

When `--train_inpainting` is set, sampling at checkpoints uses the inpainting pipeline. To provide a reference image for the masked regions, add an `--img` directive to your prompt file:

```
a photo of a cat sitting on a sofa
--img /path/to/reference.jpg
--w 512 --h 512 --d 42
```

Note that the image path cannot contain the text ` --`, as the prompt parser will take that as the start of an argument. The script will load the reference image, generate a random mask, and run the inpainting pipeline so that sample images reflect the actual inpainting task. If the specified image file does not exist, that sample is skipped with a warning rather than aborting training.

Prompt files without `--img` still work — they produce unconditional samples using the standard pipeline (no inpainting conditioning).

Note that sample images with large areas of procedural masks, which will be generated by the sampler, may fail to resolve as expected by the prompt. This is likely due the resolution of the masked pattern being interpreted as an image feature during sampling. Real world use with limited separate masked regions plus the use of feathering limit this behavior.

<details>
<summary>日本語</summary>
`--train_inpainting` が設定されている場合、チェックポイントでのサンプリングにインペインティングパイプラインが使用されます。マスク領域の参照画像を指定するには、プロンプトファイルに `--img` ディレクティブを追加します。

```
a photo of a cat sitting on a sofa
--img /path/to/reference.jpg
--w 512 --h 512 --d 42
```

画像パスに ` --` という文字列を含めることはできません（プロンプトパーサーが引数の開始として解釈するため）。スクリプトは参照画像を読み込み、ランダムなマスクを生成し、インペインティングパイプラインを実行することで、サンプル画像が実際のインペインティングタスクを反映したものになります。指定した画像ファイルが存在しない場合、学習を中断せずに警告を出してそのサンプルをスキップします。

`--img` のないプロンプトファイルも引き続き機能します — 標準パイプライン（インペインティング条件なし）を使用した通常のサンプルが生成されます。

プロシージャルマスクの面積が大きいサンプル画像では、プロンプトの内容が期待通りに反映されない場合があります。これは、マスクパターンの解像度がサンプリング中に画像の特徴として解釈されるためと考えられます。実際の使用では、マスク領域を限定し、フェザリングを適用することでこの挙動を抑制できます。
</details>

## Notes / 注意事項

- Inpainting training is compatible with LoRA, DreamBooth, fine-tuning, and textual inversion.
- The mask is applied at latent resolution (1/8 of image resolution), so very fine details at mask boundaries may be imprecise.
- For best results, start from a pre-existing inpainting checkpoint. Standard checkpoints are supported but will require more training steps to converge.
- When training SDXL, use `--gradient_checkpointing` and a memory-efficient optimizer (e.g. `Adafactor`) to reduce VRAM usage.

<details>
<summary>日本語</summary>

- インペインティング学習は LoRA、DreamBooth、ファインチューニング、テクスチャルインバージョンと互換性があります。
- マスクは画像解像度の 1/8 の潜在変数解像度で適用されるため、マスク境界の非常に細かいディテールは不正確になる場合があります。
- 最良の結果を得るには、既存のインペインティングチェックポイントから学習を開始することをお勧めします。標準チェックポイントもサポートされていますが、収束までにより多くのステップが必要になります。
- SDXL を学習する場合は、`--gradient_checkpointing` とメモリ効率の良いオプティマイザー（例: `Adafactor`）を使用して VRAM 使用量を削減してください。
</details>
