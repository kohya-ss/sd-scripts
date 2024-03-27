## リポジトリについて
Stable Diffusionの学習、画像生成、その他のスクリプトを入れたリポジトリです。

[README in English](./README.md) ←更新情報はこちらにあります

GUIやPowerShellスクリプトなど、より使いやすくする機能が[bmaltais氏のリポジトリ](https://github.com/bmaltais/kohya_ss)で提供されています（英語です）のであわせてご覧ください。bmaltais氏に感謝します。

以下のスクリプトがあります。

* DreamBooth、U-NetおよびText Encoderの学習をサポート
* fine-tuning、同上
* LoRAの学習をサポート
* 画像生成
* モデル変換（Stable Diffision ckpt/safetensorsとDiffusersの相互変換）

## 使用法について

* [学習について、共通編](./docs/train_README-ja.md) : データ整備やオプションなど
    * [データセット設定](./docs/config_README-ja.md)
* [SDXL学習](./docs/train_SDXL-en.md) （英語版）
* [DreamBoothの学習について](./docs/train_db_README-ja.md)
* [fine-tuningのガイド](./docs/fine_tune_README_ja.md):
* [LoRAの学習について](./docs/train_network_README-ja.md)
* [Textual Inversionの学習について](./docs/train_ti_README-ja.md)
* [画像生成スクリプト](./docs/gen_img_README-ja.md)
* note.com [モデル変換スクリプト](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windowsでの動作に必要なプログラム

Python 3.10.6およびGitが必要です。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

## Windows環境でのインストール

スクリプトはPyTorch 2.1.2でテストしています。PyTorch 2.0.1、1.12.1でも動作すると思われます。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

PowerShellを使う場合、通常の（管理者ではない）PowerShellを開き以下を順に実行します。

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config
```

コマンドプロンプトでも同一です。

注：`bitsandbytes==0.43.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6` は `requirements.txt` に含まれるようになりました。他のバージョンを使う場合は適宜インストールしてください。

この例では PyTorch および xfomers は2.1.2／CUDA 11.8版をインストールします。CUDA 12.1版やPyTorch 1.12.1を使う場合は適宜書き換えください。たとえば CUDA 12.1版の場合は `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` および `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121` としてください。

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

## アップグレード

新しいリリースがあった場合、以下のコマンドで更新できます。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

コマンドが成功すれば新しいバージョンが使用できます。

## 謝意

LoRAの実装は[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を基にしたものです。感謝申し上げます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

