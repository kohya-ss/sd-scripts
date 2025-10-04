# SDXL LoRA テキストエンコーダー設定オプション

SDXL の LoRA 学習では ViT-L (Text Encoder 1) と BiG-G (Text Encoder 2) の 2 系統が同時に利用されます。今回追加したオプションを使うと、学習対象となるテキストエンコーダーを選択したり、それぞれに個別の学習率を割り当てたりできます。いずれも既定値は従来と同じで、オプションを指定しなければ両方のテキストエンコーダーが同じ学習率で学習されます。

## 追加された引数

### `--network_te_train_targets`
- 形式: `--network_te_train_targets {te1|te2} [{te1|te2} ...]`
- `te1` は ViT-L、`te2` は BiG-G を指します。
- 未指定時は両方を学習します。
- `te2` を指定した場合は SDXL の 2 本目のテキストエンコーダーが読み込まれている必要があります。SD1.x/SD2.x など 1 本だけの場合に `te2` を指定するとエラーになります。
- `--network_train_unet_only` が指定されているときは無効です。

### `--text_encoder_lr1`, `--text_encoder_lr2`
- それぞれ Text Encoder 1 / 2 の学習率。未指定時は `--text_encoder_lr`、さらに未指定なら `--learning_rate` を使用します。
- 対象外のテキストエンコーダーに対して値を指定しても無視されます。例: `--network_te_train_targets te2` のときに `--text_encoder_lr1` を指定しても警告を出してスキップします。
- 値に `0` を指定するとそのテキストエンコーダーの LoRA パラメータは最適化対象から除外されます。

## 使用例

```bash
# Text Encoder 1 のみを学習し、学習率は共通値 (learning_rate) を使用
python sdxl_train_network.py \
  --network_module networks.lora \
  --network_te_train_targets te1 \
  --train_data_dir <datasets> ...

# 両方学習するが Text Encoder 1 を低い学習率に抑える
python sdxl_train_network.py \
  --network_te_train_targets te1 te2 \
  --text_encoder_lr1 5e-6 \
  --text_encoder_lr2 1e-5 \
  --text_encoder_lr 1e-5 ...

# TOML 設定ファイルの例
[train]
network_te_train_targets = ["te2"]
text_encoder_lr2 = 3.0e-6
```

## 注意点
- 既定挙動を変えないように設計されています。オプションを指定しなければ従来通り両方のテキストエンコーダーが同じ学習率で更新されます。
- LoRA ネットワーク以外 (DyLoRA / LoRA-FA など) でも同じオプションが利用できます。
- `text_encoder_lr1` / `text_encoder_lr2` を指定していても、該当するテキストエンコーダーを学習対象から外すと自動的に無視されます。
