# SDXLトークン分割ツール（sdxl_tokenize.py）

## ツールの説明
SDXL の2つのテキストエンコーダー（TE1/TE2）で、入力タグ文字列がどのようにトークン分割されるかを表示します。  
さらに、コア語の前後に1～数文字を付け足して「単文字トークンを含まない」条件で、任意の最小トークン数（既定2）以上の候補を探索できます。

## 全オプション一覧（モード別）
**共通（どのモードでも有効）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--tokenizer-cache-dir` | トークナイザのキャッシュディレクトリ | なし |
| `--add-special-tokens` | BOS/EOS などの特殊トークンを含める | 無効 |

**`--text` モード（単純分割表示）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--text` | 解析する入力文字列 | なし |

**`--search-core` モード（候補探索）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--search-core` | 探索対象のコア語（例: `sample`） | なし |
| `--search-side` | 追加文字の位置（`prefix`/`suffix`/`both`） | `both` |
| `--search-min-add` | 追加文字数の最小（`0` の場合は追加なしも探索） | `1` |
| `--search-max-add` | 追加文字数の最大 | `3` |
| `--search-min-tokens` | 探索条件として必要な最小トークン数 | `2` |
| `--search-alphabet` | 追加文字に使う文字集合 | `abcdefghijklmnopqrstuvwxyz` |
| `--search-limit` | 表示する候補数 | `20` |
| `--search-either` | TE1/TE2 のどちらかが条件を満たせば採用 | 無効 |

注意:
- `--search-core` を指定すると探索モードになり、`--text` は無視されます。
- 探索モードの条件は「単文字トークンを含まない」かつ「`--search-min-tokens` で指定したトークン数以上」です（既定 `2`）。
- `--search-either` を付けない場合は **TE1/TE2 両方**で条件を満たす候補のみ出力します。

## 使い方の例1: `--text` で分割を調べる
```bash
python tools/sdxl_tokenize.py --text "character_a,unique_01"
```
TE1/TE2 それぞれの `boundary`（`|`区切り）とトークン一覧が表示されます。

## 使い方の例2: `--search-core` で候補探索
```bash
python tools/sdxl_tokenize.py --search-core "sample" --search-side both --search-min-add 0 --search-max-add 3
```
`sample` の前後に 1～3 文字を付けた候補を探索し、条件に合う最短候補を表示します。
TE1/TE2 どちらかだけ条件を満たせば良い場合は `--search-either` を付けてください。

## 使い方の例3: `--search-min-tokens` で条件を変更
```bash
python tools/sdxl_tokenize.py --search-core "sample" --search-side both --search-min-add 0 --search-max-add 3 --search-min-tokens 3
```
`sample` の前後に付与した文字列で、`2` ではなく`3`トークン以上かつ単文字トークンを
含まない候補を探索します。
