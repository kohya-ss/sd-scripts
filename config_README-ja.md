`--config` で渡すことができる設定ファイルに関する説明です。

現在は DreamBooth の手法向けのデータセットの設定のみに対応しています。
fine tuning の手法に関わる設定及びデータセットに関わらない設定には未対応です。

## 概要

設定ファイルを渡すことにより、ユーザが細かい設定を行えるようにします。

* 複数のデータセットを指定できるようになります。
    * 例えば `resolution` や `keep_tokens` をデータセットごとに設定して、それらを混合して学習できます。

`--config` オプションは `--train_data_dir`, `--reg_data_dir` オプションとは併用不可です。

設定ファイルの形式は JSON か TOML を利用できます。
記述のしやすさを考えると TOML を利用するのがオススメです。

## データセットに関する設定

データセットに関する設定として登録可能なアイテムを説明します。

* `general`
    * 全データセットに適用されるオプションを指定します。
* `dataset`
    * 特定のデータセットに適用されるオプションを指定します。
* `dataset.subset`
    * データセット内の特定のサブセットのオプションを指定します。
    * 学習データのディレクトリの登録はここで行います。
        * ディレクトリごとにクラストークンや繰り返し回数を設定できるようにするためにサブセットとして記述する仕様にしています。

各アイテムは指定可能なオプションが以下のように決まっています。

| オプション名 | general | dataset | dataset.subset |
| ---- | ---- | ---- | ---- |
| `batch_size` | o | o | - |
| `bucket_no_upscale` | o | o | - |
| `bucket_reso_steps` | o | o | - |
| `cache_latents` | o | o | o |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_extension` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |
| `class_tokens` | - | - | o |
| `color_aug` | o | o | o |
| `enable_bucket` | o | o | - |
| `face_crop_aug_range` | o | o | o |
| `flip_aug` | o | o | o |
| `keep_tokens` | o | o | o |
| `is_reg` | - | - | o |
| `max_bucket_reso` | o | o | - |
| `min_bucket_reso` | o | o | - |
| `num_repeats` | o | o | o |
| `image_dir` | - | - | o（必須） |
| `random_crop` | o | o | o |
| `resolution` | o | o | - |
| `shuffle_caption` | o | o | o |

コマンドライン引数と共通のオプションの説明は割愛します。
他の README を参照してください。

ここでは設定ファイル特有のオプションのみ説明します。

* `batch_size`
    * バッチサイズを指定します。コマンドライン引数の `--train_batch_size` と同等です。
* `class_tokens`
    * クラストークンを設定します。例えば `sks girl` などを指定します。
    * 画像と対応する caption ファイルが存在しない場合にのみ学習時に使われます。判定は画像ごとに行います。
* `is_reg`
    * サブセットが正規化用かどうかを指定します。デフォルトは false です。
* `num_repeats`
    * サブセットの画像の繰り返し回数を指定します。
* `image_dir`
    * 画像が入ったディレクトリパスを指定します。画像はディレクトリ直下に置かれている必要があります。

## 設定ファイルの例

[samples](./samples/config_sample.toml) に例を載せているので、そちらを参照してください。
