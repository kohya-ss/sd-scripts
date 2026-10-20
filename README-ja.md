# sd-scripts

[English](./README.md) / [日本語](./README-ja.md)

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー](#スポンサー)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [更新履歴](#更新履歴)
    - [サポートモデル](#サポートモデル)
    - [機能](#機能)
- [ドキュメント](#ドキュメント)
    - [学習ドキュメント（英語および日本語）](#学習ドキュメント英語および日本語)
    - [その他のドキュメント](#その他のドキュメント)
    - [旧ドキュメント（日本語）](#旧ドキュメント日本語)
- [AIコーディングエージェントを使う開発者の方へ](#aiコーディングエージェントを使う開発者の方へ)
- [Windows環境でのインストール](#windows環境でのインストール)
    - [Windowsでの動作に必要なプログラム](#windowsでの動作に必要なプログラム)
    - [インストール手順](#インストール手順)
    - [requirements.txtとPyTorchについて](#requirementstxtとpytorchについて)
    - [xformersのインストール（オプション）](#xformersのインストールオプション)
- [Linux/WSL2環境でのインストール](#linuxwsl2環境でのインストール)
    - [DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）](#deepspeedのインストール実験的linuxまたはwsl2のみ)
- [アップグレード](#アップグレード)
    - [PyTorchのアップグレード](#pytorchのアップグレード)
- [謝意](#謝意)
- [ライセンス](#ライセンス)

</details>

## はじめに

Stable Diffusion等の画像生成モデルの学習、モデルによる画像生成、その他のスクリプトを入れたリポジトリです。

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 更新履歴

- **Version 0.12.2 (2026-10-20):**
    - LoRA 学習向けに、H2D-only block swap（`--block_swap_h2d_only`）を追加しました（Musubi Tuner からの移植）。凍結されたベースモデルの重みを CPU に置いたまま Host→Device 方向にのみコピーすることで、従来の block swap にあった Device→Host の転送が不要になり、学習速度が向上します。FLUX.1 / Chroma、Lumina Image 2.0、HunyuanImage-2.1、Anima、MOSS-Image で使用できます（SD3 は対象外）。`--gradient_checkpointing` が必須です。詳細は[ドキュメント](./docs/block_swap.md)をご覧ください。[PR #2467](https://github.com/kohya-ss/sd-scripts/pull/2467)
        - `--block_swap_ring_size`（デフォルト 2）でストリーミングするブロック用の GPU バッファ数を指定できます。`--use_pinned_memory_for_block_swap` を指定すると CPU 側のコピーを pinned メモリに置き、転送が高速になります。
        - 開発者向け: 各モデルの `enable_block_swap` はデバイスの代わりに `BlockSwapConfig` を受け取るようになりました。また offloader の `submit_move_blocks` は Musubi Tuner と同じ `submit_move_blocks_forward` に改名されています。
    - MOSS-Image で、複数のキャプション候補（`enable_multiline_caption` と区切り行）を持つ画像でも Text Encoder の出力をキャッシュできるようになりました。全候補をエンコードして一つのキャッシュファイルに保存し、ステップごとにランダムに一つを選びます。[PR #2466](https://github.com/kohya-ss/sd-scripts/pull/2466)
        - 処理済みのキャプション文字列もキャッシュファイルに保存されるため、キャプション（や `caption_prefix` / `caption_suffix`）を変更すると自動的にキャッシュが作り直されます。バージョン 0.12.1 で作成したキャッシュファイルは、候補が一つの画像ではそのまま使えます。
        - キャッシュのサイズとキャッシュ時間は候補数倍になります。他のモデルでは、複数候補を持つ画像があると引き続き Text Encoder 出力をキャッシュできません。
        - `tools/cache_text_encoder_outputs.py` は、複数候補に対応していないモデルで複数候補があるとエラーで停止するようになりました（以前は先頭の候補だけを黙ってキャッシュしていました）。

- **Version 0.12.1 (2026-10-18):**
    - [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS。Qwen3.5-9B テキストエンコーダーと FLUX.2 VAE を使う 5B の DiT）の LoRA 学習に対応しました。[PR #2458](https://github.com/kohya-ss/sd-scripts/pull/2458)、[PR #2459](https://github.com/kohya-ss/sd-scripts/pull/2459)、[PR #2460](https://github.com/kohya-ss/sd-scripts/pull/2460)、[PR #2461](https://github.com/kohya-ss/sd-scripts/pull/2461)
        - `moss_image_train_network.py` で DiT の LoRA（LoHa / LoKr も可）と、オプションでテキストエンコーダーの LoRA を学習できます。公式の diffusers 形式の分割ファイルと、ComfyUI 向けの単一ファイルのチェックポイントを読み込めます。詳細は[ドキュメント](./docs/moss_image_train_network.md)をご覧ください。
        - fp8 の代わりに、ベース重みの ConvRot INT8 量子化（DiT は `--convrot_int8`、テキストエンコーダーは `--convrot_int8_text_encoder`。Musubi Tuner からの移植）が使えます。ComfyUI の事前量子化済み `*_int8_convrot.safetensors` は自動判別されます。融合 int8 カーネルには `triton`（Windows では `triton-windows`）が必要です。
        - DiT の LoRA は変換なしで ComfyUI で使用できます。LoRA を適用した推論用に `moss_image_minimal_inference.py` も用意しています。
        - `tools/cache_latents.py` と `tools/cache_text_encoder_outputs.py` は `--moss_image` で MOSS-Image に対応しています。
        - LoRA モジュールは新規の汎用モジュール `networks.lora_generic` です（正規表現によるモジュール選択、`network_reg_dims` / `network_reg_lrs` など `networks.lora_anima` と同じ）。今後、他のアーキテクチャの基盤にする予定です。
    - 改行を含むキャプションをそのまま使うためのサブセットオプション `enable_multiline_caption`（`--enable_multiline_caption`）を追加しました。Anima、Lumina、HunyuanImage など LLM ベースの Text Encoder を持つモデル向けです。
        - `multiline_caption_separator`（デフォルト `---===---`）で始まる行で、一つの画像に対する複数のキャプション候補を区切ることができ、ステップごとにランダムに一つが選ばれます。fine tuning 方式のメタデータファイルでは `caption` に文字列のリストも指定できます。詳細は[データセット設定のドキュメント](./docs/config_README-ja.md#改行を含むキャプションと複数のキャプション候補enable_multiline_caption--true-の場合)をご覧ください。
        - `.txt` のサンプルプロンプトファイルで、`\n`（バックスラッシュ + n）が改行に変換されるようになりました。
        - 複数行キャプションで `caption_prefix` / `caption_suffix` が 1 行目 / 最終行にしか付かなかった問題を修正しました（`enable_wildcard` 有効時は prefix が 1 行目、suffix が最終行にのみ付き、無効時は suffix が消えていました）。
    - Text Encoder の出力をキャッシュする場合（`--cache_text_encoder_outputs`）に、`caption_prefix` / `caption_suffix`、`secondary_separator`、複数行キャプションの 1 行目への切り詰めが適用されず、キャプションがそのままエンコードされていた問題を修正しました。[Issue #2453](https://github.com/kohya-ss/sd-scripts/issues/2453)
        - これらのオプションを `--cache_text_encoder_outputs_to_disk` と併用している場合は、既存のキャッシュファイル（`*_te.npz` など）を削除し、処理後のキャプションで作り直してください。
        - `enable_wildcard` はキャッシュされた出力には適用できません（キャプションはそのままエンコードされます）。この場合に警告を表示するようになりました。
        - 独自のキャッシュ戦略を実装している場合: `TextEncoderOutputsCachingStrategy.cache_batch_outputs` に、処理後のキャプションを渡す `captions` 引数が追加されています。
    - `--lr_scheduler_type` でカスタムの学習率スケジューラを指定した場合に、sd-scripts が計算した総学習ステップ数と warmup ステップ数を、`num_training_steps` と `num_warmup_steps` としてスケジューラに渡すようにしました。[Issue #2444](https://github.com/kohya-ss/sd-scripts/issues/2444)
        - スケジューラのコンストラクタがこの名前の引数を持つ場合のみ渡されます。`--lr_scheduler_args` で明示的に指定した値が優先されます。異なる名前の引数（`total_steps` など）を使うスケジューラには影響しませんので、従来通り `--lr_scheduler_args` で指定してください。
        - `num_training_steps` を持つカスタムスケジューラを `--lr_scheduler_args` で指定せずに使っていた場合、これまではスケジューラのデフォルト値で動作していましたが、今後は実際の学習ステップ数が使われますのでご注意ください。同様に、`num_warmup_steps` を持つスケジューラには `--lr_warmup_steps` が渡されるようになります（従来はエラーになっていました）。

- **Version 0.12.0 (2026-09-24):**
    - Windows on ARM64（NVIDIA RTX Spark PC など）に対応しました。[PR #2430](https://github.com/kohya-ss/sd-scripts/pull/2430)、[PR #2431](https://github.com/kohya-ss/sd-scripts/pull/2431)、[PR #2433](https://github.com/kohya-ss/sd-scripts/pull/2433)
        - `opencv-python` がオプションになり（未インストール時は Pillow/NumPy による代替実装を使用）、`requirements.txt` が Windows ARM64 用 wheel のあるパッケージを自動的に選択するようになりました。詳細は[OpenCVなしでのインストール／Windows on ARM64について](#opencvなしでのインストールwindows-on-arm64について)をご覧ください。
        - `requirements.txt` の `transformers`、`schedulefree`、`safetensors` を Windows ARM64 用 wheel が提供されているバージョンに更新しました。
    - `requirements.txt` の依存関係を更新しました: `transformers` 4.57.6 → 5.5.4、`diffusers` 0.32.1 → 0.40.0、`accelerate` 1.6.0 → 1.15.0、`huggingface-hub` 0.34.3 → 1.32.0。[PR #2436](https://github.com/kohya-ss/sd-scripts/pull/2436)
        - 主にセキュリティ上の保守のための更新です（`transformers` 4.x 系と `diffusers` 0.38 未満には修正が提供されなくなっています）。以前のバージョンのライブラリでもこのリリースは動作しますので、すぐに更新する必要はありませんが、なるべく早めに `pip install --upgrade -r requirements.txt` を実行することをお勧めします。
        - `diffusers` 0.40 は PyTorch 2.6 以降を必要とします（sd-scripts はすでに PyTorch 2.6.0 以降を必要としています）。CI は PyTorch 2.6.0 と 2.8.0 でテストするようになりました。
        - `transformers` 5.x の `CLIPTokenizer` は、オリジナルの CLIP トークナイザが行っていた `ftfy` によるテキスト正規化（曲がった引用符の直線化、全角文字の半角化など）を行わなくなりました。sd-scripts 側で同じ正規化を行うようにしたため、トークナイズ結果は従来と変わりません。
        - `tests/local` のローカル回帰テストにより、Text Encoder の出力、VAE の出力、ノイズスケジューラが従来のバージョンと同一であることを確認しています。
        - `diffusers` 0.40 は `.to(dtype)` のたびに "There are modules in AutoencoderKL that should be kept in float32: [] ..." という誤った警告を出します（リストが空でも警告する diffusers 側のバグ）。sd-scripts ではリストが空の場合にこの警告を抑制しています。
    - `transformers` 5.6 以降に対応し、`requirements.txt` の `transformers` を 5.17.0 に更新しました。[PR #2437](https://github.com/kohya-ss/sd-scripts/pull/2437)
        - `transformers` 5.6 で `CLIPTextModel` の内部構造が変わりました（`text_model` サブモジュールが無くなりました）。sd-scripts ではモデルをラップすることで、チェックポイントのキー、Text Encoder の LoRA の重み名（`lora_te_text_model_...`）、`text_encoder.text_model.*` によるアクセスが従来と変わらないようにしています。`transformers` 5.6 未満ではラップは行われず、何も変わりません。
        - また `transformers` 5.6 では T5（FLUX.1 / SD3 の T5-XXL、HunyuanImage の byT5）の attention の実装が SDPA に切り替わり、高速化・省メモリ化が期待できます。T5 の bf16/fp16 出力は従来のバージョンからごくわずかに変わります（T5-XXL でコサイン類似度 ≈ 0.998、fp32 に対する精度は同等）。生成画像や学習結果の細部が変わる可能性がありますが、以前のバージョンでキャッシュした Text Encoder 出力もそのまま使えます。他の Text Encoder の出力は同一です。
        - 上記と同様に、以前のバージョンの `transformers` でも動作しますが、`pip install --upgrade -r requirements.txt` での更新をお勧めします。
    - SD1.x / SD2.x / SDXL の学習向けに、OFTv2 と BOFT のネットワークモジュール（`networks.oft_v2`、`networks.boft`）を追加しました。[PR #2357](https://github.com/kohya-ss/sd-scripts/pull/2357)
        - PEFT の実装に準拠した直交変換系のアダプタです。PEFT 形式の重みも読み込めます。umisetokikaze 氏に感謝します。
        - これらのモジュールでは `--network_dim` はブロックサイズを意味します。詳細は[ドキュメント](./docs/train_network_oft_boft.md)をご覧ください。
    - FLUX.1 および Anima の LoRA 学習で、サブセットごとの timestep sampling offset（`custom_attributes.timestep_sampling.offset`）を追加しました。[PR #2401](https://github.com/kohya-ss/sd-scripts/pull/2401) okdsf 氏に感謝します。
        - データセットのサブセットごとに、timestep のサンプリング分布を低ノイズ側または高ノイズ側へ偏らせることができます。詳細は[ドキュメント](./docs/timestep_sampling_offset.md)をご覧ください。
    - `--show_timesteps` 使用時に offset を適用した timestep の分布を確認できる `--show_timesteps_offset` を追加しました。[PR #2410](https://github.com/kohya-ss/sd-scripts/pull/2410)
        - `shift` / `flux_shift` の timestep sampling における offset の挙動もドキュメントに記載しました。
    - SD1.x / SD2.x 用の古い画像生成スクリプト `gen_img_diffusers.py` を削除しました。しばらく前から動作しておらず（リファクタリングで削除された関数に依存していました）、実験的な CLIP / VGG16 guidance を除くすべての機能は `gen_img.py` でサポートされています。代わりに `gen_img.py` をお使いください（[gen_img_README-ja.md](./docs/gen_img_README-ja.md) をご覧ください）。ファイルは以前のリリースから取得できます。[PR #2439](https://github.com/kohya-ss/sd-scripts/pull/2439)
    - 学習中のサンプル画像生成の `--sample_sampler`、および `gen_img.py` / `sdxl_gen_img.py` の `--sampler` で、`dpmsolver` と `dpmsingle` を指定すると最近のバージョンの `diffusers` でエラーになる問題を修正しました。[PR #2438](https://github.com/kohya-ss/sd-scripts/pull/2438)
        - `lms` / `k_lms` サンプラーには `requirements.txt` に含まれない `scipy` パッケージが必要です。`scipy` が未インストールの場合、（最初のサンプル生成時ではなく）起動時に分かりやすいエラーを表示するようにしました。使用する場合は `pip install scipy` を実行してください。

### サポートモデル

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**
* **Anima**
* **MOSS-Image**

### 機能

* LoRA学習
* fine-tuning（DreamBooth）：HunyuanImage-2.1以外のモデル
* Textual Inversion学習：SD/SDXL
* インペインティングモデル学習：SD1.5およびSDXL
* 画像生成
* その他、モデル変換やタグ付け、LoRAマージなどのユーティリティ

## ドキュメント

### 学習ドキュメント（英語および日本語）

日本語は折りたたまれているか、別のドキュメントにあります。

* [LoRA学習の概要](./docs/train_network.md)
* [データセット設定](./docs/config_README-ja.md) / [英語版](./docs/config_README-en.md)
* [高度な学習オプション](./docs/train_network_advanced.md)
* [OFTv2 / BOFT学習](./docs/train_network_oft_boft.md)
* [SDXL学習](./docs/sdxl_train_network.md)
* [SD3学習](./docs/sd3_train_network.md)
* [FLUX.1学習](./docs/flux_train_network.md)
* [LUMINA学習](./docs/lumina_train_network.md)
* [HunyuanImage-2.1学習](./docs/hunyuan_image_train_network.md)
* [Anima学習](./docs/anima_train_network.md)
* [MOSS-Image学習](./docs/moss_image_train_network.md)
* [Fine-tuning](./docs/fine_tune.md)
* [Textual Inversion学習](./docs/train_textual_inversion.md)
* [ControlNet-LLLite学習](./docs/train_lllite_README-ja.md) / [英語版](./docs/train_lllite_README.md)
* [Anima向けControlNet-LLLite学習ガイド](./docs/anima_train_control_net_lllite.md)
* [Validation](./docs/validation.md)
* [マスク損失学習](./docs/masked_loss_README-ja.md) / [英語版](./docs/masked_loss_README.md)
* [インペインティング学習](./docs/inpainting_training.md)

### その他のドキュメント

* [画像生成スクリプト](./docs/gen_img_README-ja.md) / [英語版](./docs/gen_img_README.md)
* [WD14 Taggerによる画像タグ付け](./docs/wd14_tagger_README-ja.md) / [英語版](./docs/wd14_tagger_README-en.md)

### 旧ドキュメント（日本語）

* [学習について、共通編](./docs/train_README-ja.md) : データ整備やオプションなど
* [DreamBoothの学習について](./docs/train_db_README-ja.md)

## AIコーディングエージェントを使う開発者の方へ

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt:

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so they won't be committed to the repository.

このリポジトリでは、AIコーディングエージェント（例：Claude、Geminiなど）がプロジェクトのコンテキストやコーディング標準を理解できるようにするための推奨プロンプトを提供しています。

それらを使用するには、プロジェクトディレクトリに設定ファイルを作成して明示的に有効にする必要があります。

**簡単なセットアップ手順:**

1.  プロジェクトルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリの推奨プロンプトをインポートします。

    ```markdown
    @./.ai/claude.prompt.md
    ```

    またはGeminiの場合:

    ```markdown
    @./.ai/gemini.prompt.md
    ``` 
3.  インポート行の下に、独自の指示を追加できます（例：`常に日本語で応答してください。`）。

この方法により、エージェントに与える指示を各開発者が管理しつつ、リポジトリの推奨コンテキストを活用できます。`CLAUDE.md` および `GEMINI.md` は `.gitignore` に登録されているため、リポジトリにコミットされることはありません。

## Windows環境でのインストール

### Windowsでの動作に必要なプログラム

Python 3.10.xおよびGitが必要です。

- Python 3.10.x: https://www.python.org/downloads/windows/ からWindows installer (64-bit)をダウンロード
- git: https://git-scm.com/download/win から最新版をダウンロード

Python 3.11.x、3.12.xでも恐らく動作します（未テスト）。

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

### インストール手順

PowerShellを使う場合、通常の（管理者ではない）PowerShellを開き以下を順に実行します。

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

コマンドプロンプトでも同一です。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

注：`bitsandbytes`、`prodigyopt`、`lion-pytorch` は `requirements.txt` に含まれています。

この例ではCUDA 12.4版をインストールします。異なるバージョンのCUDAを使用する場合は、適切なバージョンのPyTorchをインストールしてください。たとえばCUDA 12.1版の場合は `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121` としてください。

accelerate configの質問には以下のように答えてください。（bf16で学習する場合、最後の質問にはbf16と答えてください。）

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`のGPUが使われます。）

### requirements.txtとPyTorchについて

PyTorchは環境によってバージョンが異なるため、requirements.txtには含まれていません。前述のインストール手順を参考に、環境に合わせてPyTorchをインストールしてください。

スクリプトはPyTorch 2.6.0でテストしています。PyTorch 2.6.0以降が必要です。

RTX 50シリーズGPUの場合、PyTorch 2.8.0とCUDA 12.8/12.9を使用してください。`requirements.txt`はこのバージョンでも動作します。

### OpenCVなしでのインストール／Windows on ARM64について

`opencv-python` は `requirements.txt` に含まれていますが、学習・データセット処理パイプラインが利用している OpenCV 機能は限定的です（主に `cv2.resize`、`cv2.cvtColor`、およびデバッグ用の `cv2.imshow`）。`opencv-python` がインストールされていない場合、Pillow と NumPy による軽量な代替実装（`library/_cv2_stub`）が自動的に `cv2` として登録されるため、既存のスクリプトはそのまま動作します。OpenCV の大きなインストールを避けたい場合は、requirements のインストール後にアンインストールしてください：

```bash
pip uninstall opencv-python
```

Windows on ARM64（たとえば NVIDIA RTX Spark PC など）では `opencv-python` のビルド済み wheel が提供されていないため、`requirements.txt` の環境マーカーにより自動的にスキップされます。通常どおり `pip install --upgrade -r requirements.txt` でインストールできます。同じ理由で、このプラットフォームでは `tensorboard` の代わりに `tensorboardX` がインストールされます（TensorBoard 2.x が依存する `grpcio` に Windows ARM64 用 wheel がないため）。`--log_with tensorboard` でのログ出力は `tensorboardX` 経由でそのまま動作します。ログの閲覧は別のマシンの TensorBoard で行ってください。

以下に注意してください：

- OpenCV を含むデフォルトのインストールが推奨される経路です。代替実装は、データセット処理がデフォルトで使う `INTER_AREA` と `INTER_LINEAR` のリサイズを NumPy で OpenCV と同じ計算で再現しているため、学習結果は丸め誤差の範囲で一致しますが、OpenCV より低速です（2400 万画素の画像 1 枚あたり 0.1 秒程度）。`INTER_CUBIC` / `INTER_LANCZOS4` は Pillow を経由するため、わずかに結果が異なります。
- 次のツールは実際の `opencv-python` を必要とし、未インストール時は明確なメッセージで終了します：`tools/canny.py`、`tools/detect_face_rotate.py`、および `gen_img.py` / `sdxl_gen_img.py` の ControlNet `canny` プリプロセッサ。
- データセット確認時の `cv2.imshow` は、OpenCV が無い場合 Pillow 標準のビューア（`PIL.Image.show`）で表示され、`cv2.waitKey` はターミナルでの `input()` 待ちに置き換わります（1枚ずつ確認できます）。

### xformersのインストール（オプション）

xformersをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

必要に応じてCUDAバージョンを変更してください。一部のGPUアーキテクチャではxformersが利用できない場合があります。

## Linux/WSL2環境でのインストール

LinuxまたはWSL2環境でのインストール手順はWindows環境とほぼ同じです。`venv\Scripts\activate` の部分を `source venv/bin/activate` に変更してください。

※NVIDIAドライバやCUDAツールキットなどは事前にインストールしておいてください。

### DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）

DeepSpeedをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install deepspeed==0.16.7
```

## アップグレード

新しいリリースがあった場合、以下のコマンドで更新できます。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

コマンドが成功すれば新しいバージョンが使用できます。

### PyTorchのアップグレード

PyTorchをアップグレードする場合は、[Windows環境でのインストール](#windows環境でのインストール)のセクションの`pip install`コマンドを参考にしてください。

## 謝意

LoRAの実装は[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を基にしたものです。感謝申し上げます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
