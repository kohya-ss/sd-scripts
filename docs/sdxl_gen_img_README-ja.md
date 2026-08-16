# sdxl_gen_img.py 使い方

`sdxl_gen_img.py` は、SDXLモデルで画像生成するためのコマンドラインスクリプトです。

SDXL checkpoint、VAE、LoRA、Textual Inversion、ControlNet-LLLite、img2img、Highres Fixなどに対応しています。

## 基本の生成

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\sample `
  --xformers `
  --bf16 `
  --W 1024 --H 1024 `
  --steps 30 `
  --sampler euler_a `
  --scale 7 `
  --prompt "1girl, looking at viewer --n lowres, worst quality"
```

主な指定:

- `--ckpt`: SDXLモデルのcheckpointまたはDiffusers形式のディレクトリ。
- `--outdir`: 画像の出力先。
- `--W`, `--H`: 生成サイズ。未指定時は `1024x1024`。
- `--steps`: サンプリングステップ数。
- `--sampler`: サンプラー。
- `--scale`: CFG scale。
- `--prompt`: プロンプト。
- `--fp16`, `--bf16`: 推論精度。SDXLではVRAM節約のため `--bf16` が候補になります。
- `--xformers`, `--sdpa`, `--diffusers_xformers`: attention高速化・省メモリ化。

## プロンプトファイルから生成

`--from_file` で、1行1プロンプトのファイルを読み込めます。

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\batch `
  --xformers `
  --bf16 `
  --W 1024 --H 1024 `
  --steps 30 `
  --sampler euler_a `
  --scale 7 `
  --batch_size 2 `
  --vae_batch_size 1 `
  --images_per_prompt 1 `
  --from_file prompts.txt
```

`prompts.txt` の例:

```text
masterpiece, best quality, 1girl, smile --n lowres, worst quality --d 1001
masterpiece, best quality, 1girl, angry --n lowres, worst quality --d 1002
masterpiece, best quality, 1girl, full body --n lowres, worst quality --w 832 --h 1216 --d 1003
```

空行と `#` で始まる行は無視されます。

## プロンプト内オプション

プロンプト文字列の中で、`--w 1024` のような短縮オプションを指定できます。`--from_file` と組み合わせると、行ごとにサイズやseedなどを変えられます。

- `--n <text>`: ネガティブプロンプト。
- `--w <int>` / `--h <int>`: 生成サイズ。
- `--ow <int>` / `--oh <int>`: SDXL conditioning用のoriginal width/height。
- `--nw <int>` / `--nh <int>`: ネガティブ条件側のoriginal width/height。
- `--ct <int>` / `--cl <int>`: crop top / crop left。
- `--s <int>`: steps。
- `--d <seed>`: seed。`--d 1,2,3` のように複数指定できます。
- `--l <float>`: CFG scale。
- `--nl <float|none>`: negative scale。
- `--t <float>`: img2img strength。
- `--c <text>`: CLIP系ガイド用のclip prompt。
- `--am <float[,float...]>`: 追加ネットワーク、LoRA等の強度。値の個数をN/2N/3NにするとTE・U-Netを個別指定できます。
- `--dsd1`, `--dst1`, `--dsd2`, `--dst2`, `--dsr`: Deep Shrinkの上書き。
- `--glt`, `--glr`, `--gle`, `--gls`, `--glsn`, `--glus`: Gradual Latentの上書き。

行内オプションが違うプロンプトは、内部で別batchとして処理される場合があります。比較生成では、比較したい行のサイズ、steps、scale、seedを揃えると扱いやすくなります。

## SDXL conditioning

SDXLでは生成サイズのほかに、conditioning用のoriginal sizeやcrop座標を指定できます。

- `--original_width`, `--original_height`
- `--original_width_negative`, `--original_height_negative`
- `--crop_top`, `--crop_left`

プロンプト内では `--ow`, `--oh`, `--nw`, `--nh`, `--ct`, `--cl` で上書きできます。

## LoRAを使う

LoRAは追加ネットワークとして指定します。

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\lora `
  --xformers `
  --bf16 `
  --network_module networks.lora `
  --network_weights path\to\lora.safetensors `
  --network_mul 1.0 `
  --prompt "1girl, smile --n lowres, worst quality"
```

複数LoRA:

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\multi_lora `
  --xformers `
  --bf16 `
  --network_module networks.lora networks.lora `
  --network_weights path\to\lora_a.safetensors path\to\lora_b.safetensors `
  --network_mul 1.0 0.8 `
  --prompt "1girl, smile --n lowres, worst quality"
```

プロンプト内の `--am` で、行ごとにLoRA倍率だけを変えられます。

```text
1girl, smile --n lowres, worst quality --d 100 --am 0.5
1girl, smile --n lowres, worst quality --d 100 --am 1.0
```

複数LoRAの場合:

```text
1girl, smile --n lowres, worst quality --d 100 --am 1.0,0.0
1girl, smile --n lowres, worst quality --d 100 --am 0.0,1.0
```

### TE・U-NetのLoRA強度を分ける

`--network_mul` は、読み込んだLoRA数を `N` として、指定した数値の個数で意味が変わります。

| 数値の個数 | 意味 |
|---:|---|
| `N`個以下 | 従来どおり、LoRAごとの共通強度。CLIで不足したLoRAは1.0 |
| `2N`個 | LoRAごとに `TE, U-Net` の順 |
| `3N`個 | LoRAごとに `TE1, TE2, U-Net` の順 |

LoRAが1個の場合、従来の共通強度は次の指定です。

```powershell
--network_mul 0.8
```

このときTE1、TE2、U-Netはすべて0.8です。

TEを1.0、U-Netを0.5に分ける場合:

```powershell
--network_mul 1.0 0.5
```

TE1を1.0、TE2を0.25、U-Netを0.5に分ける場合:

```powershell
--network_mul 1.0 0.25 0.5
```

LoRAが2個の場合、値はLoRAごとにまとめて並べます。次の例ではLoRA Aが `TE=1.0 / U-Net=0.5`、LoRA Bが `TE=0.8 / U-Net=0.4` です。

```powershell
--network_module networks.lora networks.lora `
--network_weights path\to\lora_a.safetensors path\to\lora_b.safetensors `
--network_mul 1.0 0.5 0.8 0.4
```

TE1/TE2も分ける場合は、LoRAごとに3個ずつ並べます。

```powershell
--network_mul 1.0 0.25 0.5 0.8 1.0 0.4
```

この例の解釈:

```text
LoRA A: TE1=1.0, TE2=0.25, U-Net=0.5
LoRA B: TE1=0.8, TE2=1.0, U-Net=0.4
```

`--from_file` の行内 `--am` も同じN/2N/3N規則です。行内指定では数値をカンマで区切ります。

```text
nkzt, 1girl, solo --d 100 --am 1.0,0.5
nkzt, 1girl, solo --d 100 --am 1.0,0.25,0.5
```

複数LoRAをロードしている場合、`N` はその行で有効なLoRA数ではなく、ロード済みLoRAスロットの総数です。GUI workerは非選択スロットを0に展開するため、GUI利用時にこの個数を意識する必要はありません。

注意:

- 2N/3N指定は、推論時のLoRA差分だけをTE・U-Net別に変更します。学習率や学習済み重みは変更しません。
- TE1/TE2分離は、SDXLの2つのText Encoderグループを公開しているLoRA networkで使用できます。`networks.lora` と `networks.lora_lbw` をサポートします。
- 2N/3N指定は `--network_merge`、`--network_merge_n_models`、`--network_pre_calc` と併用できません。
- LBWとの併用時は、ここで指定した強度に各ブロックのLBWが掛かります。LBW値そのものは変更されません。
- 1を超える強度や負の強度も指定できますが、画像が不安定になる可能性があります。
- PNGメタデータには、解決後の `network-te1-strength`、`network-te2-strength`、`network-unet-strength` が保存されます。

## fork独自: SDXL LoRA Block Weight

このforkでは、`sdxl_gen_img.py` の画像生成時にSDXL LoRA Block Weightを指定するための `--network_lbw` オプションを追加しています。

学習用の `networks.lora` には手を入れず、生成専用の `networks.lora_lbw` と組み合わせて使用します。

使用例:

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\lbw `
  --xformers `
  --bf16 `
  --network_module networks.lora_lbw `
  --network_weights path\to\lora.safetensors `
  --network_mul 1.0 `
  --network_lbw XLMLT1 `
  --prompt "1girl, smile --n lowres, worst quality"
```

直接12個の数値を指定することもできます。

```powershell
--network_lbw "1,1,0,0,1,1,0.6,1,1,1,0,1"
```

複数LoRAに別々のLBWを指定する例:

```powershell
python sdxl_gen_img.py `
  --ckpt path\to\sdxl_model.safetensors `
  --outdir outputs\lbw_multi `
  --xformers `
  --bf16 `
  --network_module networks.lora_lbw networks.lora_lbw `
  --network_weights path\to\lora_a.safetensors path\to\lora_b.safetensors `
  --network_mul 1.0 1.0 `
  --network_lbw XLMIDD XLMLT1 `
  --prompt "1girl, smile --n lowres, worst quality"
```

内蔵プリセット:

- `ALL`: `1,1,1,1,1,1,1,1,1,1,1,1`
- `NONE`: `0,0,0,0,0,0,0,0,0,0,0,0`
- `XLMIDD`: `1,0,1,1,1,1,1,1,1,0,0,0`
- `XLMLT1`: `1,1,0,0,1,1,0.6,1,1,1,0,1`

SDXL LoRA Block Weightの12枠:

| 位置 | ブロック |
|---:|---|
| 1 | BASE |
| 2 | IN04 |
| 3 | IN05 |
| 4 | IN07 |
| 5 | IN08 |
| 6 | MID |
| 7 | OUT0 |
| 8 | OUT1 |
| 9 | OUT2 |
| 10 | OUT3 |
| 11 | OUT4 |
| 12 | OUT05 |

注意:

- `--network_lbw` を使う場合は `--network_module networks.lora_lbw` を指定してください。
- `--network_module networks.lora` に `--network_lbw` を組み合わせるとエラーになります。
- `--network_lbw` を指定しない場合は、従来どおり `networks.lora` を使えます。
- 現時点ではプロンプト行内の `--lbw` 指定には対応していません。

## img2img / inpainting

- `--image_path`: img2img元画像。ファイルまたはフォルダを指定できます。
- `--strength`: img2img strength。未指定時は `0.8`。
- `--mask_path`: inpainting用マスク。白い部分が処理対象です。
- `--use_original_file_name`: img2img時に元画像名を出力ファイル名に含めます。
- `--sequential_file_name`: `im_000001.png` のような連番出力にします。

img2imgでは基本的に元画像サイズが使われ、32の倍数に丸められます。

## Highres Fix

Highres Fixは、最初に小さめの画像を生成し、その画像を元にimg2imgする機能です。

- `--highres_fix_scale`: 1st stageのサイズ倍率。最終1024で1stを512にするなら `0.5`。
- `--highres_fix_steps`: 1st stage steps。既定値は `28`。
- `--highres_fix_strength`: 1st stageのimg2img strength。省略時は `--strength` と同じ。
- `--highres_fix_save_1st`: 1st stage画像も保存。
- `--highres_fix_latents_upscaling`: latentで拡大。
- `--highres_fix_upscaler`: upscaler module。例: `tools.latent_upscaler`。
- `--highres_fix_upscaler_args`: upscalerへ渡す `key=value`。
- `--highres_fix_disable_control_net`: 2nd stageでControlNetを無効化。

## ControlNet-LLLite

SDXL向けのControlNet-LLLiteを使用できます。

- `--control_net_lllite_models`: LLLiteモデルを指定します。
- `--guide_image_path`: conditioning imageを指定します。
- `--control_net_multipliers`: LLLiteの倍率。
- `--control_net_ratios`: 適用するstep比率。

プリプロセスは行われません。Canny用なら事前にCanny画像を用意してください。

## Deep Shrink

Deep Shrinkを使用できます。

- `--ds_depth_1`
- `--ds_timesteps_1`
- `--ds_depth_2`
- `--ds_timesteps_2`
- `--ds_ratio`

プロンプト内では `--dsd1`, `--dst1`, `--dsd2`, `--dst2`, `--dsr` で上書きできます。

## Gradual Latent

Gradual Latent hires fixを使用できます。

- `--gradual_latent_timesteps`
- `--gradual_latent_ratio`
- `--gradual_latent_ratio_step`
- `--gradual_latent_every_n_steps`
- `--gradual_latent_s_noise`
- `--gradual_latent_unsharp_params`

プロンプト内では `--glt`, `--glr`, `--gle`, `--gls`, `--glsn`, `--glus` で上書きできます。

## 出力

画像はPNGで保存されます。

通常のファイル名は次の形式です。

```text
im_<timestamp>_<index>_<seed>.png
```

PNG metadataには、プロンプト、steps、sampler、scale、seed、negative promptなどが保存されます。

