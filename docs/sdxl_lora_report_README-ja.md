# SDXL LoRA Report

`sdxl_lora_report` は、SDXL LoRAの比較用画像をまとめて生成し、ブラウザで見やすいHTMLレポートを作るための補助ツールです。

主な目的は、LoRAの差分確認、LBW(LoRA Block Weight)の効き方確認、複数LoRAの重ね掛け比較を、同じprompt/seed条件で効率よく行うことです。

## できること

- 1つ以上のLoRA条件を指定して画像を一括生成する。
- 同じprompt、同じseedで、LoRAだけ違う画像を並べて比較する。
- 1つの比較条件に複数LoRAを入れて、LoRAの重ね掛けを比較する。
- LoRAごとに `strength` と `lbw` を指定する。
- 必要に応じてLoRAなしの `baseline` 画像も生成する。
- 生成結果を1つの出力フォルダにまとめる。
- HTMLレポートでLoRA、prompt、seedの表示切り替えや画像サイズ変更を行う。

## ファイル構成

```text
sdxl_lora_report_gui.py
  PySide6 GUI。通常はこちらを使う。

sdxl_lora_report_cui.py
  config JSONを読んでジョブを展開し、workerを呼び、metadataとHTMLを作る。

sdxl_lora_report_worker.py
  1回の sdxl_gen_img.py 起動で複数LoRA条件を切り替えながら生成するworker。

lora_report_samples/
  CUI用のサンプルconfigとpromptファイル。
```

通常の実行経路は以下です。

```text
sdxl_lora_report_gui.py
  -> sdxl_lora_report_cui.py
    -> sdxl_lora_report_worker.py
      -> sdxl_gen_img.py
```

## GUIの使い方

PySide6が必要です。`requirements.txt` に追加されていますが、既存環境を壊したくない場合は、必要なvenvに手動で `PySide6` だけ入れてください。

起動例:

```powershell
python sdxl_lora_report_gui.py
```

画面上部で以下を指定します。

- `Model`: SDXLモデルファイル。`.safetensors` または `.ckpt` をドラッグ&ドロップできます。
- `Output root`: レポートの出力先フォルダ。フォルダをドラッグ&ドロップできます。
- `Prompt file`: prompt一覧のテキストファイル。`.txt` をドラッグ&ドロップできます。
- `Run name`: 出力フォルダ名に使う識別名。

左側の `LoRA assets` に `.safetensors` のLoRAをドラッグ&ドロップします。

中央の `Comparison conditions` が、レポート上で比較される条件です。

- `Make single conditions`: 選択中のLoRAから、単体LoRA条件をまとめて作ります。
- `Add condition`: 空の比較条件を作ります。
- `Add selected LoRA`: 選択中のLoRAを、選択中の比較条件に追加します。
- `Move up` / `Move down`: 選択中の比較条件を上下に移動します。この順番がHTMLレポート上のLoRA条件の並び順になります。
- 1つの条件にLoRAが1個なら単体LoRA、2個以上なら重ね掛け条件です。
- 条件内のLoRA行で `Strength(s)` と `LBW` を編集できます。
- LoRA行を選んで `Edit strengths` を押すか、Strength(s)セルをダブルクリックすると、`Common`、`TE / U-Net`、`TE1 / TE2 / U-Net` のモードを意味付き入力欄で選べます。
- Strength(s)セルへ直接入力する場合は、セルを選んで `F2` を押します。1個なら共通強度、2個なら `TE, U-Net`、3個なら `TE1, TE2, U-Net` の順で、複数値はカンマ区切りです。

Strength(s)の例:

```text
0.8              # TE1=0.8, TE2=0.8, U-Net=0.8
1.0, 0.5         # TE1=1.0, TE2=1.0, U-Net=0.5
1.0, 0.25, 0.5   # TE1=1.0, TE2=0.25, U-Net=0.5
```

左側の `Default strengths` にも同じ3モードがあります。初期値は従来どおり `Common = 0.8` です。`Default strengths` と `Default LBW` は、LoRAファイルをassets一覧へ追加した時ではなく、`Make single conditions` または `Add selected LoRA` で比較条件へ登録した時点の値が適用されます。このため、ファイル追加後にdefaultを変更してから条件を作っても新しい値が反映されます。

右側で生成設定を指定します。

- `Width` / `Height`
- `Steps`
- `Sampler`
- `Scale`
- `Batch size`
- `Precision`
- `Attention`
- `Extra args`
- `Seeds`
- `Include baseline`

`Include baseline` をオンにすると、LoRAなし画像も同じprompt/seedで生成します。

`Precision` は `fp32 / none`、`bf16`、`fp16` から選びます。`bf16` を選ぶと `--bf16`、`fp16` を選ぶと `--fp16` が `common_args` に入ります。

`Attention` は `none`、`sdpa`、`xformers` から選びます。`sdpa` と `xformers` は通常どちらか一方だけを使うため、GUIでは同時指定できない選択式にしています。

`Extra args` には、GUIに専用欄がない追加オプションをスペース区切りで指定できます。
ただし `--network_merge` / `--network_merge_n_models` は使えません。LoRAをモデルへマージしてしまうと、レポート用のLoRA条件切り替えができなくなるためです。

`Batch size` を2以上にすると高速化できる場合がありますが、同じseedでも生成結果が微妙に変わることがあります。LoRA差分を厳密に比較したい場合は `Batch size = 1` を推奨します。

### Queue

生成に時間がかかる場合は、現在の設定をキューに入れてからまとめて実行できます。

- `Run report`: 現在の設定をすぐに生成します。
- `Add to queue`: 現在の設定をキューに追加します。キューに入った内容はスナップショットなので、その後GUI上の設定を変えても既存キューには影響しません。
- `Run queue`: `Waiting` 状態のキューを上から順番に実行します。
- `Stop after current`: 現在実行中のジョブが終わったところでキュー実行を止めます。
- `Cancel running`: 現在実行中のキュージョブを中断します。
- `Remove selected`: 選択したキュー項目を削除します。
- `Clear done`: 完了済みのキュー項目を削除します。
- `Load selected into setup`: 選択したキュー項目の設定をGUI上部、生成設定、LoRA条件へ読み戻します。少し編集して再実行したい場合に使います。
- `Open selected report`: 完了したキュー項目のHTMLレポートを開きます。

キューは `.tmp/queue/queue_state.json` に保存され、次回GUI起動時に復元されます。

## 前回設定の復元

GUI実行時には `.tmp/lora_report_gui_last.json` が作られます。

これはGUIからCUIへ渡す実行用configですが、次回GUI起動時に以下の設定を復元するためにも使われます。

- Model
- width
- height
- steps
- sampler
- scale
- batch_size
- common_args

LoRA条件やprompt fileまでは自動復元しません。実験条件を勝手に持ち越す事故を避けるためです。

`.tmp/` は `.gitignore` に入っています。実モデルパスや個人環境のLoRAパスが入るため、コミットしないでください。

## Prompt File

promptファイルは1行1promptです。

`.txt` と `.tsv` に対応しています。`.txt` でも先頭の有効行がタブ区切りヘッダーで、`prompt` 列を含む場合はTSVとして自動判定します。

Excelなどの表計算ソフトで編集する場合は、`.tsv` がおすすめです。promptはカンマを多く含むため、CSVよりタブ区切りのほうが扱いやすいです。

TSVのヘッダー:

```text
id	prompt	negative	width	height
```

TSV例:

```text
id	prompt	negative	width	height
standing_smile	1girl, standing, smile	low quality, worst quality	1024	1024
sitting_smile	1girl, sitting, smile	low quality, worst quality	1024	1024
```

TSVでは `prompt` 列が必須です。`id`、`negative`、`width`、`height` は省略できます。

`width` と `height` を指定する場合は、SDXLのUNet内部サイズ不一致を避けるため64の倍数にしてください。たとえば `896`、`960`、`1024` などです。64の倍数でない場合は、生成開始前にエラーになります。

`.txt` では以下の形式も使えます。

単純な書き方:

```text
1girl, standing, smile
1girl, sitting, smile
```

詳細指定:

```text
prompt_id | prompt | negative prompt | width | height
```

例:

```text
standing_smile | 1girl, standing, smile | low quality, worst quality | 1024 | 1024
```

空行と `#` で始まる行は無視されます。

## CUIの使い方

GUIを使わずにconfig JSONから実行することもできます。

サンプル:

```powershell
python sdxl_lora_report_cui.py --config lora_report_samples\lora_report_sample.json --dry-run
```

本番実行:

```powershell
python sdxl_lora_report_cui.py --config path\to\config.json
```


## Config JSON

サンプルは `lora_report_samples/lora_report_sample.json` です。

主な項目:

```json
{
  "output_root": "../../lora_reports",
  "run_name": "sample_lora_compare",
  "prompt_file": "lora_report_prompts_sample.txt",
  "sdxl_gen_img": {
    "ckpt": "D:/models/sdxl_model.safetensors",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "sampler": "euler_a",
    "scale": 7.0,
    "batch_size": 1,
    "images_per_prompt": 1,
    "common_args": ["--bf16", "--sdpa"]
  },
  "seeds": {
    "values": [12345],
    "random_count": 0
  },
  "include_baseline": true,
  "loras": []
}
```

LoRA単体条件:

```json
{
  "id": "sample_xlmlt1",
  "name": "Sample LoRA XLMLT1",
  "path": "D:/loras/sample_lora.safetensors",
  "strength": 1.0,
  "lbw": "XLMLT1"
}
```

LoRA重ね掛け条件:

```json
{
  "id": "sample_stack",
  "name": "Sample LoRA stack",
  "items": [
    {
      "name": "character",
      "path": "D:/loras/sample_character.safetensors",
      "strength": [1.0, 0.5],
      "lbw": "XLMLT1"
    },
    {
      "name": "style",
      "path": "D:/loras/sample_style.safetensors",
      "strength": 0.5,
      "lbw": "ALL"
    }
  ]
}
```

`strength` は従来どおり数値1個も使用できます。配列が2個なら `TE, U-Net`、3個なら `TE1, TE2, U-Net` として扱います。

LBWを使う条件では、その条件内のすべてのLoRA itemに `lbw` を指定してください。

通常LoRAとして比較したい条件では `lbw` を省略できます。その場合、worker内では `ALL` 相当として扱われます。

## HTML Report

出力先には日時つきフォルダが作られます。

```text
output_root/
  20260617_120000_run_name/
    report.html
    blind_report.html
    metadata.json
    config.json
    prompts.txt
    prompts.parsed.json
    images/
    worker/
```

`report.html` では以下ができます。

- X軸/Y軸の入れ替え
- LoRA条件、prompt、seedの表示/非表示切り替え
- 画像表示サイズのスライダー変更
- 画像クリックで元画像表示

`blind_report.html` はブラインドテスト用の別HTMLです。

- 同じprompt/seedの画像をLoRA名を伏せて表示
- HTMLを開くたびに各prompt/seed内の画像順をランダム化
- 各prompt/seedごとにBestを1つ選択
- `Reveal / 答え合わせ` でLoRA名と得票数を表示

## 実装コンセプト

このツールは、`sdxl_gen_img.py` の生成機能を直接大きく改造せず、比較レポート用の薄い制御層として作っています。

重要な考え方:

- GUIは生成ロジックを持たない。
- GUIはconfig JSONを作り、CUIを起動するだけにする。
- CUIはconfigを正規化し、prompt/seed/LoRA条件をジョブへ展開する。
- workerは `sdxl_gen_img.py --from_file --sequential_file_name` を1回だけ起動し、各prompt行の `--am` でLoRA強度を切り替える。
- LoRAの読み込み回数とモデルロード回数を減らし、比較条件が増えても扱いやすくする。

workerは全LoRA条件で必要なLoRAをスロット化します。

同じ `(module, path, lbw)` のLoRAは1つのスロットとしてまとめられます。各画像生成時には、条件に応じて `--am` の強度を変えます。TE/U-Net分離を含む条件では、workerが全スロットを自動的に2Nまたは3N形式へ展開します。

例:

```text
LoRA A only:  --am 0.8,0.0
LoRA B only:  --am 0.0,0.8
LoRA A+B:     --am 0.8,0.8
baseline:     --am 0.0,0.0
```

2スロットのうちLoRA Aだけを `TE=1.0 / U-Net=0.5` で使う場合、2N形式へ展開されます。

```text
LoRA A only:  --am 1.0,0.5,0.0,0.0
```

いずれかのLoRAがTE1/TE2分離を使う場合は、他の共通指定やTE/U-Net指定も3N形式へ展開されます。

これにより、単体LoRA、複数LoRA重ね掛け、baselineを同じ生成プロセス内で扱えます。

## 制約

- `width` と `height` は64の倍数である必要があります。
- 現在のworkerは `images_per_prompt=1` を前提にしています。GUIではこの値を表示せず、常に1としてconfigを生成します。
- SDXL LoRA LBW利用を主目的にしているため、network moduleは `networks.lora_lbw` のみをサポート対象にします。`networks.dylora` など他のnetwork moduleとの混在は非対応です。
- GUIの前回復元対象はモデルと生成設定だけです。LoRA条件やprompt fileは自動復元しません。
- `.tmp/` は個人環境のパスを含むためコミット対象外です。

## 今後の開発メモ

次に拡張するなら、以下が候補です。

- GUIのLoRA条件ツリーへのドラッグ&ドロップ操作をさらに直感的にする。
- LBWプリセット管理画面を追加する。
- 1つのLoRAに対して複数LBWを自動展開するLBW sweep機能を追加する。
- strength sweep機能を追加する。
- promptごとの画像サイズや個別negative prompt編集をGUI上で行えるようにする。
- レポートに条件メモや評価コメント欄を追加する。
- 生成済みレポートをGUIから開く履歴機能を追加する。
