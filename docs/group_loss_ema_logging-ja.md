# グループ別Loss(EMA)ログ機能（Phase1）

`sdxl_train_network.py`（`train_network.py` 共通ループ）に追加された、学習中の `loss` をグループ単位で可視化する機能です。

## 概要

- 目的:
  - 複数キャラ同時学習時に、キャラごとの学習進み具合の偏りを観測する
- 機能:
  - subsetごとに設定した `group` で、学習lossのオンラインEMAを集計
  - stepログCSV（全step記録）とepochサマリCSVを出力

## 制約条件

- `--group_loss_log` は **batch_size=1 専用**です。
  - データセット定義の `batch_size` が 1 以外ならエラーで停止します。
  - 実行中バッチサイズが 1 以外でもエラーで停止します。
- 想定利用は DreamBooth の `class_tokens` 方式です（最低限この方式をサポート）。
- 分散学習時（DDP）は **main process のみCSV出力**します（rank0のローカルバッチに基づくログ）。
- `skip_grad_norm` で更新がスキップされたstepは、EMA集計にもCSVにも含みません。
- `loss` が `NaN` / `inf` のstepは、EMA集計にもCSVにも含みません。

## dataset_config.toml 拡張

`[[datasets.subsets]]` に以下の任意キーを追加できます。

- `group = "character_a"`
  - グループ識別子（文字列）

`group` 未指定（または空文字）のsubsetは、ログ上 `__ungrouped__` として扱われます。

## 参考情報（重複サブセットとキャッシュ）

- 同一 `[[datasets]]` 内で、DreamBooth方式の同一 `image_dir` 重複subsetは **後続が無視**されます。
  - この重複判定は `image_dir` ベースです。`class_tokens` が違っていても別扱いにはなりません。
- 同一 `[[datasets]]` 内で、FineTuning方式の同一 `metadata_file` 重複subsetも後続が無視されます。
- 同じ画像フォルダを別設定で使いたい場合は、`[[datasets]]` を分ければ併用可能です。
- ただし、`[[datasets]]` を分けても同じ実画像パスを再利用する場合は、次のディスクキャッシュ系オプションは避けてください。
  - `--cache_latents_to_disk`
  - `--cache_text_encoder_outputs_to_disk`
  - 同じキャッシュファイルパスを共有して衝突・上書きが発生する可能性があります。
- 参考:
  - `--cache_latents`（メモリキャッシュのみ）は通常この制約に当たりません。

## CLIオプション一覧

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--group_loss_log` | `False` | グループ別Loss(EMA)ログ機能を有効化 |
| `--group_loss_ema_beta <float>` | `0.98` | EMA係数（`ema = ema*beta + loss*(1-beta)`） |
| `--group_loss_log_every_n_steps <int>` | `100` | stepログCSVのバッファを書き出す間隔（global step）。記録自体は全stepで行う |
| `--group_loss_epoch_summary` | `False` | epoch末サマリCSVを追記出力 |

## 出力ファイル

`output_dir` 配下に出力されます。`output_name` 未指定時は `last` が使われます。

- stepログ:
  - `group_loss_logs+<output_name>.csv`
- epochサマリ（`--group_loss_epoch_summary` 有効時のみ）:
  - `group_loss_epoch+<output_name>.csv`

## CSV列定義

### stepログCSV

ヘッダ:

`global_step,epoch,group,subset_index,loss,ema_loss_group,count_group,timestep,bucket_reso`

- `global_step`: optimizer更新単位のstep
- `epoch`: 1始まり
- `group`: subsetに設定したgroup（未指定は `__ungrouped__`）
- `subset_index`: 全dataset通しのsubset識別子
- `loss`: そのstepのloss
- `ema_loss_group`: そのgroupのEMA値
- `count_group`: そのgroupの有効step累計
- `timestep`: diffusion timestep
- `bucket_reso`: バケット解像度（`WxH`）

### epochサマリCSV

ヘッダ:

`epoch,group,ema_loss_end,count_epoch,mean_loss_epoch`

- `ema_loss_end`: そのepoch終了時点のEMA
- `count_epoch`: そのepoch内の有効step数
- `mean_loss_epoch`: そのepoch内のgroup平均loss

## CLI指定例

### 最小例（stepログのみ）

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/out \
  --output_name sample_lora \
  --group_loss_log
```

### 全step記録 + 100stepごとにflush + epochサマリ

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/out \
  --output_name sample_lora \
  --group_loss_log \
  --group_loss_ema_beta 0.98 \
  --group_loss_log_every_n_steps 100 \
  --group_loss_epoch_summary
```

## dataset toml 記載例

### 1) キャラごとにgroupを付与

```toml
[general]
enable_bucket = true

[[datasets]]
resolution = [1024, 1024]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\datasets\example\character_a"
  class_tokens = "character_a"
  num_repeats = 20
  group = "character_a"

  [[datasets.subsets]]
  image_dir = "D:\datasets\example\character_b"
  class_tokens = "character_b"
  num_repeats = 20
  group = "character_b"
```

### 2) 解像度違いの `[[datasets]]` をまたいで同一groupに集約

```toml
[general]
enable_bucket = true

[[datasets]]
resolution = [720, 720]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\datasets\example\small\character_a"
  class_tokens = "character_a"
  num_repeats = 20
  group = "character_a"

[[datasets]]
resolution = [1024, 1024]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\datasets\example\large\character_a"
  class_tokens = "character_a"
  num_repeats = 10
  group = "character_a"
```

この例では、`group=character_a` のログが2つの `[[datasets]]` をまたいで合算されます。
