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

- **次のリリースに含まれる予定の更新:** 次のリリースに含まれる予定の主な変更点は以下の通りです。リリース前の変更点は予告なく変更される可能性があります。
    - `requirements.txt` の依存関係を更新しました: `transformers` 4.57.6 → 5.5.4、`diffusers` 0.32.1 → 0.40.0、`accelerate` 1.6.0 → 1.15.0、`huggingface-hub` 0.34.3 → 1.32.0。[PR #2436](https://github.com/kohya-ss/sd-scripts/pull/2436)
        - 主にセキュリティ上の保守のための更新です（`transformers` 4.x 系と `diffusers` 0.38 未満には修正が提供されなくなっています）。更新後は `pip install --upgrade -r requirements.txt` を実行してください。
        - `diffusers` 0.40 は PyTorch 2.6 以降を必要とします（sd-scripts はすでに PyTorch 2.6.0 以降を必要としています）。CI は PyTorch 2.6.0 と 2.8.0 でテストするようになりました。
        - `transformers` 5.6 以降はまだサポートしていません。5.6 で `CLIPTextModel` の内部構造が変わり、Text Encoder のチェックポイントおよび LoRA の重み名が変わってしまうためです。当面は 5.5.x をお使いください。
        - `transformers` 5.x の `CLIPTokenizer` は、オリジナルの CLIP トークナイザが行っていた `ftfy` によるテキスト正規化（曲がった引用符の直線化、全角文字の半角化など）を行わなくなりました。sd-scripts 側で同じ正規化を行うようにしたため、トークナイズ結果は従来と変わりません。
        - `tests/local` のローカル回帰テストにより、Text Encoder の出力、VAE の出力、ノイズスケジューラが従来のバージョンと同一であることを確認しています。
        - `diffusers` 0.40 は `.to(dtype)` のたびに "There are modules in AutoencoderKL that should be kept in float32: [] ..." という誤った警告を出します（リストが空でも警告する diffusers 側のバグ）。sd-scripts ではリストが空の場合にこの警告を抑制しています。
    - SD1.x / SD2.x 用の古い画像生成スクリプト `gen_img_diffusers.py` を削除しました。しばらく前から動作しておらず（リファクタリングで削除された関数に依存していました）、実験的な CLIP / VGG16 guidance を除くすべての機能は `gen_img.py` でサポートされています。代わりに `gen_img.py` をお使いください（[gen_img_README-ja.md](./docs/gen_img_README-ja.md) をご覧ください）。ファイルは以前のリリースから取得できます。[PR #2439](https://github.com/kohya-ss/sd-scripts/pull/2439)
    - 学習中のサンプル画像生成の `--sample_sampler`、および `gen_img.py` / `sdxl_gen_img.py` の `--sampler` で、`dpmsolver` と `dpmsingle` を指定すると最近のバージョンの `diffusers` でエラーになる問題を修正しました。[PR #2438](https://github.com/kohya-ss/sd-scripts/pull/2438)
        - `lms` / `k_lms` サンプラーには `requirements.txt` に含まれない `scipy` パッケージが必要です。`scipy` が未インストールの場合、（最初のサンプル生成時ではなく）起動時に分かりやすいエラーを表示するようにしました。使用する場合は `pip install scipy` を実行してください。
    - Windows on ARM64（NVIDIA RTX Spark PC など）に対応しました。[PR #2430](https://github.com/kohya-ss/sd-scripts/pull/2430)、[PR #2431](https://github.com/kohya-ss/sd-scripts/pull/2431)、[PR #2433](https://github.com/kohya-ss/sd-scripts/pull/2433)
        - `opencv-python` がオプションになり（未インストール時は Pillow/NumPy による代替実装を使用）、`requirements.txt` が Windows ARM64 用 wheel のあるパッケージを自動的に選択するようになりました。詳細は[OpenCVなしでのインストール／Windows on ARM64について](#opencvなしでのインストールwindows-on-arm64について)をご覧ください。
        - `requirements.txt` の `transformers`、`schedulefree`、`safetensors` を Windows ARM64 用 wheel が提供されているバージョンに更新しました。
    - SD1.x / SD2.x / SDXL の学習向けに、OFTv2 と BOFT のネットワークモジュール（`networks.oft_v2`、`networks.boft`）を追加しました。[PR #2357](https://github.com/kohya-ss/sd-scripts/pull/2357)
        - PEFT の実装に準拠した直交変換系のアダプタです。PEFT 形式の重みも読み込めます。umisetokikaze 氏に感謝します。
        - これらのモジュールでは `--network_dim` はブロックサイズを意味します。詳細は[ドキュメント](./docs/train_network_oft_boft.md)をご覧ください。
    - FLUX.1 および Anima の LoRA 学習で、サブセットごとの timestep sampling offset（`custom_attributes.timestep_sampling.offset`）を追加しました。[PR #2401](https://github.com/kohya-ss/sd-scripts/pull/2401) okdsf 氏に感謝します。
        - データセットのサブセットごとに、timestep のサンプリング分布を低ノイズ側または高ノイズ側へ偏らせることができます。詳細は[ドキュメント](./docs/timestep_sampling_offset.md)をご覧ください。
    - `--show_timesteps` 使用時に offset を適用した timestep の分布を確認できる `--show_timesteps_offset` を追加しました。[PR #2410](https://github.com/kohya-ss/sd-scripts/pull/2410)
        - `shift` / `flux_shift` の timestep sampling における offset の挙動もドキュメントに記載しました。

- **Version 0.11.1 (2026-06-16):**
    - Anima LoRA／LLLite学習でtorch.compileサポートを追加しました。[PR #2379](https://github.com/kohya-ss/sd-scripts/pull/2379)
        - 学習が20%ほど高速化されるようです。動作にはTritonやMSVCコンパイラが必要です。詳細は[ドキュメント](./docs/anima_torch_compile.md)をご覧ください。
    - 2DのみのQwen-Image VAEを追加しました。[PR #2382](https://github.com/kohya-ss/sd-scripts/pull/2382)
        - [issue #2369](https://github.com/kohya-ss/sd-scripts/issues/2369) での woct0rdho 氏の提案に基づいています。woct0rdho 氏に感謝します。
        - `--qwen_image_vae_2d` を指定すると有効になります。重みは通常版（3D版）と同じものが使用できます。
        - latentの事前キャッシュの高速化が期待できます（学習自体は変わりません）。詳細は[ドキュメント](./docs/anima_train_network.md#memory-and-speed--メモリ速度関連)をご覧ください。
    - LLLiteインペインティングモデルの学習サポートを追加しました。[PR #2378](https://github.com/kohya-ss/sd-scripts/pull/2378)
        - 詳細は[ドキュメント](./docs/anima_train_control_net_lllite.md)をご覧ください。
    - timestep samplingの設定値のログ出力、timestepsの分布の可視化を追加しました。[PR #2384](https://github.com/kohya-ss/sd-scripts/pull/2384)
        - 可視化により学習がどのようなタイムステップで行われるかを理解しやすくなります。
        - 詳細は[ドキュメント](./docs/anima_train_network.md#visualizing-the-timestep-distribution)をご覧ください。

- **Version 0.11.0 (2026-06-12):**
    - コードベースの大規模な内部リファクタリングを行い、コードベースの品質と保守性を向上させました。[PR #2372](https://github.com/kohya-ss/sd-scripts/pull/2372)
        - ユーザーの方には直接の影響が極力少なくなるよう配慮しました。詳細について、および不具合報告などは[こちらのdiscussion](https://github.com/kohya-ss/sd-scripts/discussions/2358)までお願いします。

- **Version 0.10.6 (2026-06-12):**
    - リファクタリングマージ前の安定バージョン。

- **Version 0.10.5 (2026-05-08):**
    - transformersのバージョン5以降に対応しました。[PR #2315](https://github.com/kohya-ss/sd-scripts/pull/2315) および [PR #2316](https://github.com/kohya-ss/sd-scripts/pull/2316) marcus165090-spec氏に感謝します。
        - `requirements.txt`の`transformers`のバージョンは4.xのままですが、5.xでも動作します。何らかの理由で5.xを用いる場合はdiffusersもあわせて最新バージョンにしてください。
    - Anima向けのControlNet-LLLite学習に対応しました。[PR #2317](https://github.com/kohya-ss/sd-scripts/pull/2317)
        - 詳細は[ドキュメント](./docs/anima_train_control_net_lllite.md)をご覧ください。

- **Version 0.10.4 (2026-05-07):**
    - Intel GPUの互換性を向上しました。[PR #2307](https://github.com/kohya-ss/sd-scripts/pull/2307) WhitePr氏に感謝します。
    - SD 1.5/SDXLのinpaintingモデルの学習に対応しました。[PR #2309](https://github.com/kohya-ss/sd-scripts/pull/2309) および [PR #2318](https://github.com/kohya-ss/sd-scripts/pull/2318)allanoepping氏に感謝します。
        - 詳細は[ドキュメント](./docs/inpainting_training.md)をご覧ください。

### サポートモデル

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**
* **Anima**

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
