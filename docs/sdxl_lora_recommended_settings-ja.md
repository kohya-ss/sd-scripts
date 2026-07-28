# SDXL LoRA 推奨学習設定

キャラクターの再現度を優先しつつ、過学習を抑えて柔軟に扱える LoRA を作るための設定

## 特徴

- VRAM 12GB または 16GB を想定
- batch size 1  (2以上にすると上手くいかない)
- fp16  
- rank 4～6  
- 画像数が少ない前提で dropout を高めにし、2万stepくらいじっくり学習する   
   例：
   5～10枚くらいの画像を用意し、各2～4つ程度のトリミング違いを用意して20枚くらいにし
   各 20～30 repeatにして 1 Epock 500stepくらい x 40 Epoc = 20000 stepくらい) 
- タグは独自タグのみで最小限に絞る  
- 本 fork の独自オプション群で過学習を抑えつつ柔軟性を高める  
- LoRA完成後、使用時に LoRA Block Weight で調整する前提  

## 全オプション一覧（学習コマンド）

| 設定項目 | 値 | 解説 | 補足説明 |
|---|---|---|---|
| **基本設定** |  |  |  |
| `--num_cpu_threads_per_process` | 8 | DataLoader など CPU スレッド数 | `accelerate launch` 側の指定 |
| 実行スクリプト | `sdxl_train_network.py` | SDXL LoRA 学習の本体スクリプト |  |
| `--pretrained_model_name_or_path` | `D:\train_model\xxx.safetensors` | 学習元のモデル |  |
| `--dataset_config` | `D:\train_data\dataset_config.toml` | dataset 設定ファイル | 下の TOML の例を参照 |
| `--prior_loss_weight` | 1.0 | prior loss の重み | prior 付き学習時の重み係数 |
| `--output_dir` | `..\lora_output` | 出力先ディレクトリ |  |
| `--output_name` | `loraABC` | 出力ファイル名のベース |  |
| `--learning_rate` | 3.5e-4 | UNet 側の基本学習率 | 3.5e-4か4e-4くらいが良い |
| `--max_train_epochs` | 40 | 最大エポック数 |  |
| `--optimizer_type` | AdamW8bit | 8bit AdamW |  |
| `--sdpa` | 有効 | SDPA を使う | 環境依存で有効化される --xformersが使えるならそちらでも可 |
| `--mixed_precision` | fp16 | 学習の混合精度 | fp16 前提 |
| `--save_precision` | fp16 | 保存精度 |  |
| `--seed` | 39 | 乱数シード | 何でもよいはず |
| `--save_model_as` | safetensors | 保存形式 |  |
| `--save_every_n_epochs` | 1 | エポックごとの保存間隔 |  |
| `--max_data_loader_n_workers` | 1 | DataLoader の worker 数 |  |
| **LoRA 基本設定** |  |  |  |
| `--network_module` | `networks.lora` | LoRA 実装モジュール |  |
| `--network_dim` | 4 | LoRA rank | 低 rank を基本に調整 4,5,6 あたりを使用 alphaは1(デフォルト) |
| `--network_args`（rank_dropout） | `rank_dropout=0.2` | LoRA の rank dropout | 過学習を防ぐ |
| **解像度・ノイズ・キャッシュ** |  |  |  |
| `--enable_bucket` | 有効 | バケット有効化 | TOML 側でも指定 |
| `--min_bucket_reso` | 384 | バケット最小解像度 |  |
| `--max_bucket_reso` | 1024 | バケット最大解像度 |  |
| `--noise_offset` | 0.15 | ノイズオフセット | 何となく設定 |
| `--adaptive_noise_scale` | 0.1 | 適応ノイズスケール | 何となく設定 |
| `--network_dropout` | 0.3 | LoRA dropout | 過学習を防ぐ |
| `--cache_latents` | 有効 | latent をキャッシュ | 速度・安定性向上 |
| **Text Encoder 関連** |  |  |  |
| `--text_encoder_lr` | 2e-4 | TE 共通の学習率 | 独自オプションで上書きしている |
| `--text_encoder_lr1`（独自） | 3e-4 | TE1 の学習率 | SDXL 2 系統 TE を分離 |
| `--text_encoder_lr2`（独自） | 2e-4 | TE2 の学習率 |  |
| `--downscale_freq_shift`（独自） | 有効 | 時間埋め込みの周波数ダウンスケールを 1.0 側に戻す | キャラ LoRA の identity 定着を狙う |
| `--te_mlp_fc_only`（独自） | 有効 | TE の学習対象を MLP (FC) 層のみに限定 | 語彙を固めつつ柔軟性を残す狙い |
| **安定化・平均化関連** |  |  |  |
| `--grad_norm_mode`（独自） | stable または stable_no_threshoff | skip_grad_norm の推奨プリセット | スパイクとなるstepの学習をskipして安定重視の挙動に寄せる stableの設定がいいと思っていたが、stable_no_threshoff でもいいかもしれない |
| `--rank_log`（独自） | 有効 | rank の変動や傾向を確認するためのログ出力 | 解析用のログを残すオプション 学習中の rank の挙動や偏りを後から見返したいときに役立つ |
| `--rank_log_mode`（独自） | per_module | rank_log の集計粒度 | `per_module` はモジュールごとに記録する 詳しく傾向を見たいとき向け |
| `--avg_cp`（独自） | 有効 | エポック平均（SWA/EMA 相当）を有効化 | LoRA 部分のみが対象 エポックを重みづけ平均しながら学習を進める(安定熟成狙い) |
| `--avg_cp_mode`（独自） | promote | avg_cp の動作モード | `live` は従来どおり平均重みをそのまま学習へ反映 `shadow` は raw と center を比較してログと best 保存だけ行う `promote` は比較結果が良いときだけ center を次 epoch に採用 推奨は `promote` |
| `--avg_window`（独自） | 4 | 平均に使う ckpt 数 | 窓サイズ |
| `--avg_begin`（独自） | 0.6 | 平均を開始する進行率 | 進行率 60% 以降 |
| `--avg_mode`（独自） | ema | 平均方法 | 直近ほど重くする EMA |
| `--avg_promote_pick`（独自） | fixed | promote 候補の選び方 | `promote` の `fixed` は `avg_mode` で指定した候補だけを採点して使う `best` は proxy bank 上で `ema` と `uniform` の両方を採点して良い方を選ぶ。`shadow` は常に両方を採点する |
| `--avg_shadow_bank_size`（独自） | 12 | proxy bank に保持するサンプル数 | 学習タグが多い場合は必要に応じて数を増やす |
| `--avg_save_last_candidates`（独自） | 無効 | 最終 raw / center を追加保存 | `shadow` / `promote` 専用 `<output_name>_raw.safetensors` と `<output_name>_center.safetensors` を保存する |
| `--avg_reset_stats`（独自） | `--no-avg_reset_stats` | 平均後に optimizer 統計をリセット | 推奨は `--no-avg_reset_stats` `promote` ではまず reset なしを試し、不安定なら戻す `live` も伸び鈍化が気になるなら reset なしを試す価値が高い。注: `AdamW8bit` は現行のリセット対象である `exp_avg` / `exp_avg_sq` / `exp_avg_max` を optimizer state に持たないため、実質何もリセットされない |
| `--fp16_safe_norms`（独自） | 有効 | fp16 + 小バッチで学習安定性を向上 | フェイク量子化とセットで運用 |
| **学習率スケジュール** |  |  |  |
| `--lr_scheduler` | constant_with_warmup | 学習率スケジューラ | 初期のみ 学習率をwarmup (最初に学習率0→指定値まで線形に上昇) |
| `--lr_warmup_steps` | 0.05 | 学習率warmup 比率 | 全体の 5% |

## dq_delta 関連オプション

最初は付けずに学習する方が無難なので、量子化系オプションは別表に分離

| 設定項目 | 値 | 解説 | 補足説明 |
|---|---|---|---|
| `--dq_delta_bits`（独自） | 8 | dq_delta の量子化ビット数 | フェイク量子化の粗さ 基本は8bit(スケジュールする場合は上書きされる) 処理時間が1.5倍くらいになるが上手くいけば柔軟な学習結果になる |
| `--dq_delta_granularity`（独自） | channel | dq_delta の粒度 | チャネル別に統計 |
| `--dq_delta_stat`（独自） | rms | dq_delta の統計基準 | range_mul の基準 |
| `--dq_delta_range_mul`（独自） | 3.0 | dq_delta のレンジ倍率 | 量子化レンジの広さ |
| `--dq_delta_mode`（独自） | stoch | dq_delta の丸め方式 | 確率的丸め |
| `--dq_delta_begin_after_lr_warmup`（独自） | 有効 | 学習率のwarmup完了後からdq_delta を開始 | 事前に学習率のwarmupをすることで量子化を安定させる |
| `--dq_delta_scope`（独自） | unet | dq_delta の適用対象 | U-Net のみ |
| `--dq_delta_bits_sched`（独自） | `0.0:8,0.9:10` | 進行率で bits を切替 | 終盤で刻みを細かくする 効果があるか不明　途中でbit数を増やすのと破綻しやすくなる可能性もありそう |
| `--dq_delta_log`（独自） | 有効 | dq_delta の統計ログを出力 | クリップ率等の確認用 |
| `--dq_delta_auto_range_mul`（独自） | 有効 | clip_rate を見て range_mul を自動調整 | 過/不足クリップを自動補正 データセットによっては初期値から変わらないこともある |
| `--dq_delta_auto_preset`（独自） | clip_rate_high | auto 調整のプリセット | clip_low/high を選択。`clip_rate_low_auto` はlow帯で開始し、`QErrPerClip` が高い状態が続くとmid帯へ逃がす |
| `--dq_delta_auto_init_range_mul_from_band`（独自） | 有効 | clip 帯中心から range_mul 初期値を算出 | `stat=rms` 前提 適正なrange_mulからスタートすることで安定させる |
| `--dq_delta_auto_use_raw`（独自） | 有効 | auto 判定に ema だけでなくraw も併用 | range_mulの変化をなだらかにする |

### 量子化のヒント

- 量子化はしないのが一番無難
- 最初から量子化有で学習するとディティールが弱くなりやすいが、うまく行けば柔軟性が増す
- まず量子化無で学習し、`--network_weights` でその重みを読み込んで量子化ありで追加学習するのもよい
- `clip_rate_low` が合うデータもあるが、lowを維持するための量子化誤差が重くなる場合は `--dq_delta_auto_preset clip_rate_low_auto` でmid帯へ自動退避させる選択肢がある
- `--dq_delta_bits_sched "0.0:8,0.9:10"` で前半を 8bit、最後 10% だけ 10bit にする設定は、うまくいったこともあるが基本的には不要。ずっと 8bit のままで問題ない
- `--dq_quantize_z --dq_delta_mode det` は 10% 程度高速化する。学習は破綻なくできたが、内部で行われる丸め処理の場所が変わるため結果は通常設定と同等ではなくなる。量子化による自由度アップ性能が劣るようなので不採用

## 検証メモ

### TE の warmup を長くし、途中で TE を freeze する実験

通常設定:

```bat
--lr_scheduler constant_with_warmup --lr_warmup_steps 0.05
```

実験設定:

```bat
--lr_scheduler constant_with_warmup --lr_warmup_steps 0.05 --te1_lr_warmup_steps 0.1 --te2_lr_warmup_steps 0.1 --te1_freeze_at 0.85 --te2_freeze_at 0.85
```

- TE の warmup を長くすると TE の学習が薄まり、全体として学習が弱くなる
- 通常の U-Net と同期した warmup の方がよい
- TE を後半で freeze しても、特に良い効果はなかった

## データセット設定例（dataset_config.toml）
```
キャラクタータグ二人分 stnc,tkgw
画像ごとの捨てタグ xa,xb... 画像1種類に1つ
1つのフォルダに同じ画像のトリミング違いの画像を1～4個ほど入れる
その他1girlなどのタグは付けない

[general]
enable_bucket = true
shuffle_caption = false

[[datasets]]
resolution = [720,720]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\stnc_saa\'
  class_tokens = 'stnc,xa'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\stnc_sbb\'
  class_tokens = 'stnc,xb'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\stnc_scc\'
  class_tokens = 'stnc,xc'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\stnc_sdd\'
  class_tokens = 'stnc,xd'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\stnc_see\'
  class_tokens = 'stnc,xe'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\tkgw_taa\'
  class_tokens = 'tkgw,xg'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\tkgw_tbb\'
  class_tokens = 'tkgw,xh'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\tkgw_tcc\'
  class_tokens = 'tkgw,xi'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\tkgw_tdd\'
  class_tokens = 'tkgw,xj'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\train_data\01_データ\tkgw_tee\'
  class_tokens = 'tkgw,xk'
  num_repeats = 20
  flip_aug = false

```
