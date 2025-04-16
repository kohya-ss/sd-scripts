ステータス：確認中

---

# 高度な設定: SDXL LoRA学習スクリプト `sdxl_train_network.py` 詳細ガイド

このドキュメントでは、`sd-scripts` リポジトリに含まれる `sdxl_train_network.py` を使用した、SDXL (Stable Diffusion XL) モデルに対する LoRA (Low-Rank Adaptation) モデル学習の高度な設定オプションについて解説します。

基本的な使い方については、以下のドキュメントを参照してください。

*   [LoRA学習スクリプト `train_network.py` の使い方](train_network.md)
*   [SDXL LoRA学習スクリプト `sdxl_train_network.py` の使い方](sdxl_train_network.md)

このガイドは、基本的なLoRA学習の経験があり、より詳細な設定や高度な機能を試したい熟練した利用者を対象としています。

**前提条件:**

*   `sd-scripts` リポジトリのクローンと Python 環境のセットアップが完了していること。
*   学習用データセットの準備と設定（`.toml`ファイル）が完了していること。（[データセット設定ガイド](link/to/dataset/config/doc)参照）
*   基本的なLoRA学習のコマンドライン実行経験があること。

## 1. コマンドライン引数 詳細解説

`sdxl_train_network.py` は `train_network.py` の機能を継承しつつ、SDXL特有の機能を追加しています。ここでは、SDXL LoRA学習に関連する主要なコマンドライン引数について、機能別に分類して詳細に解説します。

基本的な引数については、[LoRA学習スクリプト `train_network.py` の使い方](train_network.md#31-主要なコマンドライン引数) および [SDXL LoRA学習スクリプト `sdxl_train_network.py` の使い方](sdxl_train_network.md#31-主要なコマンドライン引数（差分）) を参照してください。

### 1.1. モデル読み込み関連

*   `--pretrained_model_name_or_path="<モデルパス>"` **[必須]**
    *   学習のベースとなる **SDXLモデル** を指定します。Hugging Face HubのモデルID、ローカルのDiffusers形式モデルディレクトリ、または`.safetensors`ファイルを指定できます。
    *   詳細は[基本ガイド](sdxl_train_network.md#モデル関連)を参照してください。
*   `--vae="<VAEパス>"`
    *   オプションで、学習に使用するVAEを指定します。SDXLモデルに含まれるVAE以外を使用する場合に指定します。`.ckpt`または`.safetensors`ファイルを指定できます。
*   `--no_half_vae`
    *   混合精度(`fp16`/`bf16`)使用時でもVAEを`float32`で動作させます。SDXLのVAEは`float16`で不安定になることがあるため、`fp16`指定時には有効にすることが推奨されます。`bf16`では通常不要です。
*   `--fp8_base` / `--fp8_base_unet`
    *   **実験的機能:** ベースモデル（U-Net, Text Encoder）またはU-NetのみをFP8で読み込み、VRAM使用量を削減します。PyTorch 2.1以上が必要です。詳細は TODO 後でドキュメントを追加 の関連セクションを参照してください (SD3の説明ですがSDXLにも適用されます)。

### 1.2. データセット設定関連

*   `--dataset_config="<設定ファイルのパス>"` 
    *   データセットの設定を記述した`.toml`ファイルを指定します。SDXLでは高解像度データとバケツ機能（`.toml` で `enable_bucket = true` を指定）の利用が一般的です。
    *   `.toml`ファイルの書き方の詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください。
    *   アスペクト比バケツの解像度ステップ(`bucket_reso_steps`)は、SDXLでは32の倍数とする必要があります。

### 1.3. 出力・保存関連

基本的なオプションは `train_network.py` と共通です。

*   `--output_dir="<出力先ディレクトリ>"` **[必須]**
*   `--output_name="<出力ファイル名>"` **[必須]**
*   `--save_model_as="safetensors"` (推奨), `ckpt`, `pt`, `diffusers`, `diffusers_safetensors`
*   `--save_precision="fp16"`, `"bf16"`, `"float"`
    *   モデルの保存精度を指定します。未指定時は学習時の精度(`fp16`, `bf16`等)で保存されます。
*   `--save_every_n_epochs=N` / `--save_every_n_steps=N`
    *   Nエポック/ステップごとにモデルを保存します。
*   `--save_last_n_epochs=M` / `--save_last_n_steps=M`
    *   エポック/ステップごとに保存する際、最新のM個のみを保持し、古いものは削除します。
*   `--save_state` / `--save_state_on_train_end`
    *   モデル保存時/学習終了時に、Optimizerの状態などを含む学習状態(`state`)を保存します。`--resume`オプションでの学習再開に必要です。
*   `--save_last_n_epochs_state=M` / `--save_last_n_steps_state=M`
    *   `state`の保存数をM個に制限します。`--save_last_n_epochs/steps`の指定を上書きします。
*   `--no_metadata`
    *   出力モデルにメタデータを保存しません。
*   `--save_state_to_huggingface` / `--huggingface_repo_id` など
    *   Hugging Face Hubへのモデルやstateのアップロード関連オプション。詳細は TODO ドキュメントを追加 を参照してください。

### 1.4. ネットワークパラメータ (LoRA)

基本的なオプションは `train_network.py` と共通です。

*   `--network_module=networks.lora` **[必須]**
*   `--network_dim=N` **[必須]**
    *   LoRAのランク (次元数) を指定します。SDXLでは32や64などが試されることが多いですが、データセットや目的に応じて調整が必要です。
*   `--network_alpha=M`
    *   LoRAのアルファ値。`network_dim`の半分程度、または`network_dim`と同じ値などが一般的です。デフォルトは1。
*   `--network_dropout=P`
    *   LoRAモジュール内のドロップアウト率 (0.0~1.0)。過学習抑制の効果が期待できます。デフォルトはNone (ドロップアウトなし)。
*   `--network_args ...`
    *   ネットワークモジュールへの追加引数を `key=value` 形式で指定します。LoRAでは以下の高度な設定が可能です。
        *   **階層別 (Block-wise) 次元数/アルファ:**
            *   U-Netの各ブロックごとに異なる`dim`と`alpha`を指定できます。これにより、特定の層の影響を強めたり弱めたりする調整が可能です。
            *   `block_dims`: U-NetのLinear層およびConv2d 1x1層に対するブロックごとのdimをカンマ区切りで指定します (SDXLでは23個の数値)。
            *   `block_alphas`: 上記に対応するalpha値をカンマ区切りで指定します。
            *   `conv_block_dims`: U-NetのConv2d 3x3層に対するブロックごとのdimをカンマ区切りで指定します。
            *   `conv_block_alphas`: 上記に対応するalpha値をカンマ区切りで指定します。
            *   指定しないブロックは `--network_dim`/`--network_alpha` または `--conv_dim`/`--conv_alpha` (存在する場合) の値が使用されます。
            *   詳細は[LoRA の階層別学習率](train_network.md#lora-の階層別学習率) (train\_network.md内、SDXLでも同様に適用可能) や実装 ([lora.py](lora.py)) を参照してください。
        *   **LoRA+:**
            *   `loraplus_lr_ratio=R`: LoRAの上向き重み(UP)の学習率を、下向き重み(DOWN)の学習率のR倍にします。学習速度の向上が期待できます。論文推奨は16。
            *   `loraplus_unet_lr_ratio=RU`: U-Net部分のLoRA+学習率比を個別に指定します。
            *   `loraplus_text_encoder_lr_ratio=RT`: Text Encoder部分のLoRA+学習率比を個別に指定します。(`--text_encoder_lr1`, `--text_encoder_lr2`で指定した学習率に乗算されます)
            *   詳細は[README](../README.md#jan-17-2025--2025-01-17-version-090)や実装 ([lora.py](lora.py)) を参照してください。
*   `--network_train_unet_only`
    *   U-NetのLoRAモジュールのみを学習します。Text Encoderの学習を行わない場合に指定します。`--cache_text_encoder_outputs` を使用する場合は必須です。
*   `--network_train_text_encoder_only`
    *   Text EncoderのLoRAモジュールのみを学習します。U-Netの学習を行わない場合に指定します。
*   `--network_weights="<重みファイル>"`
    *   学習済みのLoRA重みを読み込んで学習を開始します。ファインチューニングや学習再開に使用します。`--resume` との違いは、このオプションはLoRAモジュールの重みのみを読み込み、`--resume` はOptimizerの状態や学習ステップ数なども復元します。
*   `--dim_from_weights`
    *   `--network_weights` で指定した重みファイルからLoRAの次元数 (`dim`) を自動的に読み込みます。`--network_dim` の指定は不要になります。

### 1.5. 学習パラメータ

*   `--learning_rate=LR`
    *   全体の学習率。各モジュール(`unet_lr`, `text_encoder_lr1`, `text_encoder_lr2`)のデフォルト値となります。`1e-3` や `1e-4` などが試されることが多いです。
*   `--unet_lr=LR_U`
    *   U-Net部分のLoRAモジュールの学習率。
*   `--text_encoder_lr1=LR_TE1`
    *   Text Encoder 1 (OpenCLIP ViT-G/14) のLoRAモジュールの学習率。通常、U-Netより小さい値 (例: `1e-5`, `2e-5`) が推奨されます。
*   `--text_encoder_lr2=LR_TE2`
    *   Text Encoder 2 (CLIP ViT-L/14) のLoRAモジュールの学習率。通常、U-Netより小さい値 (例: `1e-5`, `2e-5`) が推奨されます。
*   `--optimizer_type="..."`
    *   使用するOptimizerを指定します。`AdamW8bit` (省メモリ、一般的), `Adafactor` (さらに省メモリ、SDXLフルモデル学習で実績あり), `Lion`, `DAdaptation`, `Prodigy`などが選択可能です。各Optimizerには追加の引数が必要な場合があります (`--optimizer_args`参照)。
    *   `AdamW8bit` や `PagedAdamW8bit` (要 `bitsandbytes`) が一般的です。
    *   `Adafactor` はメモリ効率が良いですが、設定がやや複雑です (相対ステップ(`relative_step=True`)推奨、学習率スケジューラは`adafactor`推奨)。
    *   `DAdaptation`, `Prodigy` は学習率の自動調整機能がありますが、LoRA+との併用はできません。学習率は`1.0`程度を指定します。
    *   詳細は[train\_util.py](train_util.py)の`get_optimizer`関数を参照してください。
*   `--optimizer_args ...`
    *   Optimizerへの追加引数を `key=value` 形式で指定します (例: `"weight_decay=0.01"` `"betas=0.9,0.999"`).
*   `--lr_scheduler="..."`
    *   学習率スケジューラを指定します。`constant` (変化なし), `cosine` (コサインカーブ), `linear` (線形減衰), `constant_with_warmup` (ウォームアップ付き定数), `cosine_with_restarts` など。`constant` や `cosine` 、 `constant_with_warmup` がよく使われます。
    *   スケジューラによっては追加の引数が必要です (`--lr_scheduler_args`参照)。
    *   `DAdaptation` や `Prodigy` などの自己学習率調整機能付きOptimizerを使用する場合、スケジューラは不要です (`constant` を指定)。
*   `--lr_warmup_steps=N`
    *   学習率スケジューラのウォームアップステップ数。学習開始時に学習率を徐々に上げていく期間です。N < 1 の場合は全ステップ数に対する割合と解釈されます。
*   `--lr_scheduler_num_cycles=N` / `--lr_scheduler_power=P`
    *   特定のスケジューラ (`cosine_with_restarts`, `polynomial`) のためのパラメータ。
*   `--max_train_steps=N` / `--max_train_epochs=N`
    *   学習の総ステップ数またはエポック数を指定します。エポック指定が優先されます。
*   `--mixed_precision="bf16"` / `"fp16"` / `"no"`
    *   混合精度学習の設定。SDXLでは `bf16` (対応GPUの場合) または `fp16` の使用が強く推奨されます。VRAM使用量を削減し、学習速度を向上させます。
*   `--full_fp16` / `--full_bf16`
    *   勾配計算も含めて完全に半精度/bf16で行います。VRAM使用量をさらに削減できますが、学習の安定性に影響する可能性があります。VRAMがどうしても足りない場合に使用します。
*   `--gradient_accumulation_steps=N`
    *   勾配をNステップ分蓄積してからOptimizerを更新します。実質的なバッチサイズを `train_batch_size * N` に増やし、少ないVRAMで大きなバッチサイズ相当の効果を得られます。デフォルトは1。
*   `--max_grad_norm=N`
    *   勾配クリッピングの閾値。勾配のノルムがNを超える場合にクリッピングします。デフォルトは1.0。`0`で無効。
*   `--gradient_checkpointing`
    *   メモリ使用量を大幅に削減しますが、学習速度は若干低下します。SDXLではメモリ消費が大きいため、有効にすることが推奨されます。
*   `--fused_backward_pass`
    *   **実験的機能:** 勾配計算とOptimizerのステップを融合し、VRAM使用量を削減します。SDXLで利用可能です。現在 `Adafactor` Optimizerのみ対応。Gradient Accumulationとは併用できません。
*   `--resume="<stateディレクトリ>"`
    *   `--save_state`で保存された学習状態から学習を再開します。Optimizerの状態や学習ステップ数などが復元されます。

### 1.6. キャッシュ機能関連

SDXLは計算コストが高いため、キャッシュ機能が効果的です。

*   `--cache_latents`
    *   VAEの出力(Latent)をメモリにキャッシュします。VAEの計算を省略でき、VRAM使用量を削減し、学習を高速化します。**注意:** 画像に対するAugmentation (`color_aug`, `flip_aug`, `random_crop` 等) は無効になります。
*   `--cache_latents_to_disk`
    *   `--cache_latents` と併用し、キャッシュ先をディスクにします。大量のデータセットや複数回の学習で特に有効です。初回実行時にディスクにキャッシュが生成され、2回目以降はそれを読み込みます。
*   `--cache_text_encoder_outputs`
    *   Text Encoderの出力をメモリにキャッシュします。Text Encoderの計算を省略でき、VRAM使用量を削減し、学習を高速化します。**注意:** キャプションに対するAugmentation (`shuffle_caption`, `caption_dropout_rate` 等) は無効になります。**また、このオプションを使用する場合、Text EncoderのLoRAモジュールは学習できません (`--network_train_unet_only` の指定が必須です)。**
*   `--cache_text_encoder_outputs_to_disk`
    *   `--cache_text_encoder_outputs` と併用し、キャッシュ先をディスクにします。
*   `--skip_cache_check`
    *   キャッシュファイルの内容の検証をスキップします。ファイルの存在確認は行われ、存在しない場合はキャッシュが生成されます。デバッグ等で意図的に再キャッシュしたい場合を除き、通常は指定不要です。

### 1.7. サンプル画像生成関連

基本的なオプションは `train_network.py` と共通です。

*   `--sample_every_n_steps=N` / `--sample_every_n_epochs=N`
    *   Nステップ/エポックごとにサンプル画像を生成します。
*   `--sample_at_first`
    *   学習開始前にサンプル画像を生成します。
*   `--sample_prompts="<プロンプトファイル>"`
    *   サンプル画像生成に使用するプロンプトを記述したファイル (`.txt`, `.toml`, `.json`) を指定します。書式は[gen\_img\_diffusers.py](gen_img_diffusers.py)に準じます。詳細は[ドキュメント](gen_img_README-ja.md)を参照してください。
*   `--sample_sampler="..."`
    *   サンプル画像生成時のサンプラー（スケジューラ）を指定します。`euler_a`, `dpm++_2m_karras` などが一般的です。選択肢は `--help` を参照してください。

### 1.8. Logging & Tracking 関連

*   `--logging_dir="<ログディレクトリ>"`
    *   TensorBoardなどのログを出力するディレクトリを指定します。指定しない場合、ログは出力されません。
*   `--log_with="tensorboard"` / `"wandb"` / `"all"`
    *   使用するログツールを指定します。`wandb`を使用する場合、`pip install wandb`が必要です。
*   `--log_prefix="<プレフィックス>"`
    *   `logging_dir` 内に作成されるサブディレクトリ名の接頭辞を指定します。
*   `--wandb_api_key="<APIキー>"` / `--wandb_run_name="<実行名>"`
    *   Weights & Biases (wandb) 使用時のオプション。
*   `--log_tracker_name` / `--log_tracker_config`
    *   高度なトラッカー設定用オプション。通常は指定不要。
*   `--log_config`
    *   学習開始時に、使用された学習設定（一部の機密情報を除く）をログに出力します。再現性の確保に役立ちます。

### 1.9. 正則化・高度な学習テクニック関連

*   `--noise_offset=N`
    *   ノイズオフセットを有効にし、その値を指定します。画像の明るさやコントラストの偏りを改善する効果が期待できます。SDXLのベースモデルはこの値で学習されているため、有効にすることが推奨されます (例: 0.0357)。元々の技術解説は[こちら](https://www.crosslabs.org/blog/diffusion-with-offset-noise)。
*   `--noise_offset_random_strength`
    *   ノイズオフセットの強度を0から指定値の間でランダムに変動させます。
*   `--adaptive_noise_scale=N`
    *   Latentの平均絶対値に応じてノイズオフセットを調整します。`--noise_offset`と併用します。
*   `--multires_noise_iterations=N` / `--multires_noise_discount=D`
    *   複数解像度ノイズを有効にします。異なる周波数成分のノイズを加えることで、ディテールの再現性を向上させる効果が期待できます。イテレーション回数N (6-10程度) と割引率D (0.3程度) を指定します。技術解説は[こちら](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)。
*   `--ip_noise_gamma=G` / `--ip_noise_gamma_random_strength`
    *   Input Perturbation Noiseを有効にします。入力(Latent)に微小なノイズを加えて正則化を行います。Gamma値 (0.1程度) を指定します。`random_strength`で強度をランダム化できます。
*   `--min_snr_gamma=N`
    *   Min-SNR Weighting Strategy を適用します。学習初期のノイズが大きいタイムステップでのLossの重みを調整し、学習を安定させます。`N=5` などが使用されます。
*   `--scale_v_pred_loss_like_noise_pred`
    *   v-predictionモデルにおいて、vの予測ロスをノイズ予測ロスと同様のスケールに調整します。SDXLはv-predictionではないため、**通常は使用しません**。
*   `--v_pred_like_loss=N`
    *   ノイズ予測モデルにv予測ライクなロスを追加します。`N`でその重みを指定します。SDXLでは**通常は使用しません**。
*   `--debiased_estimation_loss`
    *   Debiased EstimationによるLoss計算を行います。Min-SNRと類似の目的を持ちますが、異なるアプローチです。
*   `--loss_type="l1"` / `"l2"` / `"huber"` / `"smooth_l1"`
    *   損失関数を指定します。デフォルトは`l2` (MSE)。`huber`や`smooth_l1`は外れ値に頑健な損失関数です。
*   `--huber_schedule="constant"` / `"exponential"` / `"snr"`
    *   `huber`または`smooth_l1`損失使用時のスケジューリング方法。`snr`が推奨されています。
*   `--huber_c=C` / `--huber_scale=S`
    *   `huber`または`smooth_l1`損失のパラメータ。
*   `--masked_loss`
    *   マスク画像に基づいてLoss計算領域を限定します。データセット設定で`conditioning_data_dir`にマスク画像（白黒）を指定する必要があります。詳細は[マスクロスについて](masked_loss_README.md)を参照してください。

### 1.10. 分散学習・その他

*   `--seed=N`
    *   乱数シードを指定します。学習の再現性を確保したい場合に設定します。
*   `--max_token_length=N` (`75`, `150`, `225`)
    *   Text Encoderが処理するトークンの最大長。SDXLでは通常`75` (デフォルト) または `150`, `225`。長くするとより複雑なプロンプトを扱えますが、VRAM使用量が増加します。
*   `--clip_skip=N`
    *   Text Encoderの最終層からN層スキップした層の出力を使用します。SDXLでは**通常使用しません**。
*   `--lowram` / `--highvram`
    *   メモリ使用量の最適化に関するオプション。`--lowram`はColabなどRAM < VRAM環境向け、`--highvram`はVRAM潤沢な環境向け。
*   `--persistent_data_loader_workers` / `--max_data_loader_n_workers=N`
    *   DataLoaderのワーカプロセスに関する設定。エポック間の待ち時間やメモリ使用量に影響します。
*   `--config_file="<設定ファイル>"` / `--output_config`
    *   コマンドライン引数の代わりに`.toml`ファイルを使用/出力するオプション。
*   **Accelerate/DeepSpeed関連:** (`--ddp_timeout`, `--ddp_gradient_as_bucket_view`, `--ddp_static_graph`)
    *   分散学習時の詳細設定。通常はAccelerateの設定 (`accelerate config`) で十分です。DeepSpeedを使用する場合は、別途設定が必要です。

## 2. その他のTips

*   **VRAM使用量:** SDXL LoRA学習は多くのVRAMを必要とします。24GB VRAMでも設定によってはメモリ不足になることがあります。以下の設定でVRAM使用量を削減できます。
    *   `--mixed_precision="bf16"` または `"fp16"` (必須級)
    *   `--gradient_checkpointing` (強く推奨)
    *   `--cache_latents` / `--cache_text_encoder_outputs` (効果大、制約あり)
    *   `--optimizer_type="AdamW8bit"` または `"Adafactor"`
    *   `--gradient_accumulation_steps` の値を増やす (バッチサイズを小さくする)
    *   `--full_fp16` / `--full_bf16` (安定性に注意)
    *   `--fp8_base` / `--fp8_base_unet` (実験的)
    *   `--fused_backward_pass` (Adafactor限定、実験的)
*   **学習率:** SDXL LoRAの適切な学習率はデータセットや`network_dim`/`alpha`に依存します。`1e-4` ~ `4e-5` (U-Net), `1e-5` ~ `2e-5` (Text Encoders) あたりから試すのが一般的です。
*   **学習時間:** 高解像度データとSDXLモデルのサイズのため、学習には時間がかかります。キャッシュ機能や適切なハードウェアの利用が重要です。
*   **トラブルシューティング:**
    *   **NaN Loss:** 学習率が高すぎる、混合精度の設定が不適切 (`fp16`時の`--no_half_vae`未指定など)、データセットの問題などが考えられます。
    *   **VRAM不足 (OOM):** 上記のVRAM削減策を試してください。
    *   **学習が進まない:** 学習率が低すぎる、Optimizer/Schedulerの設定が不適切、データセットの問題などが考えられます。

## 3. おわりに

`sdxl_train_network.py` は非常に多くのオプションを提供しており、SDXL LoRA学習の様々な側面をカスタマイズできます。このドキュメントが、より高度な設定やチューニングを行う際の助けとなれば幸いです。

不明な点や詳細については、各スクリプトの `--help` オプションや、リポジトリ内の他のドキュメント、実装コード自体を参照してください。

---