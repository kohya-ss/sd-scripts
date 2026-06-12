# Fine-tuning Metadata File Specification / fine-tuning メタデータファイル仕様

This document specifies the metadata file format consumed by the **fine-tuning dataset** of `sd-scripts` (the dataset selected when a subset has a `metadata_file` option, equivalent to the `--in_json` command-line argument).

It describes only the **contract that the trainer reads at load time** — i.e. what `library/finetuning_dataset.py` actually parses. It is intentionally tool-agnostic: any tool, script, or AI may generate these files as long as they conform to the format below. Fields not listed here are silently ignored by the trainer.

For how to register a metadata file in a dataset configuration, see the [Dataset Configuration Guide](./config_README-en.md).

<details>
<summary>日本語</summary>

このドキュメントは、`sd-scripts` の **fine-tuning データセット**（サブセットに `metadata_file` オプションを指定したとき／コマンドライン引数 `--in_json` に相当する場合に選択されるデータセット）が読み込むメタデータファイルの形式を仕様化したものです。

ここで規定するのは、**学習スクリプトが読み込み時に解釈する契約**、すなわち `library/finetuning_dataset.py` が実際にパースする内容のみです。意図的にツール非依存としてあり、以下の形式に従ってさえいれば、任意のツール・スクリプト・AI でこれらのファイルを生成して構いません。ここに記載のないフィールドは学習スクリプトから黙って無視されます。

メタデータファイルをデータセット設定に登録する方法については、[データセット設定ガイド](./config_README-ja.md) を参照してください。

</details>

## 1. Overview / 概要

A metadata file maps each image to its caption / tags / size. Two formats are supported, selected by file extension:

* **Standard JSON** (`.json`): a single JSON object whose keys are image paths.
* **JSON Lines** (`.jsonl`): one JSON object per line, one image per line.

The trainer reads, per image, at most three pieces of information:

| Field | Type | Required | Purpose |
|---|---|---|---|
| `caption` | string | no (defaults to empty) | Training caption / conditioning text |
| `tags` | string | no | Additional tags, merged into the caption (see §4) |
| `image_size` | `[width, height]` | no (but recommended) | Bucketing without reading image files (see §5) |

Everything else in the file is ignored.

<details>
<summary>日本語</summary>

メタデータファイルは、各画像をそのキャプション／タグ／サイズに対応づけます。形式は拡張子で選択され、2 種類がサポートされます。

* **標準 JSON** (`.json`): キーを画像パスとする、単一の JSON オブジェクト。
* **JSON Lines** (`.jsonl`): 1 行 1 オブジェクト、1 行 1 画像。

学習スクリプトが画像ごとに読み取る情報は、最大で次の 3 つだけです。

| フィールド | 型 | 必須 | 用途 |
|---|---|---|---|
| `caption` | 文字列 | 任意（省略時は空文字列） | 学習キャプション／条件付けテキスト |
| `tags` | 文字列 | 任意 | 追加タグ。キャプションに連結される（§4 参照） |
| `image_size` | `[幅, 高さ]` | 任意（ただし推奨） | 画像ファイルを読まずにバケツ分けするための情報（§5 参照） |

これら以外のフィールドはすべて無視されます。

</details>

## 2. Standard JSON format / 標準 JSON 形式

The file is a single object. Each **key** is an image path; each **value** is an object with the optional fields `caption`, `tags`, `image_size`.

```json
{
  "001.png": {
    "caption": "a cat sitting on a sofa",
    "tags": "cat, sofa, indoor",
    "image_size": [1024, 768]
  },
  "002.png": {
    "caption": "a dog running on a beach"
  },
  "subdir/003.png": {
    "tags": "dog, beach, sunset",
    "image_size": [768, 1024]
  }
}
```

Notes:

* `caption`, `tags`, and `image_size` are all optional. An entry with none of them is still a valid (empty-caption) training sample.
* `tags` is a **string**, not an array. When present it is merged into the caption as described in §4.
* The file must contain at least one entry; an empty object causes the subset to be skipped with a warning.

<details>
<summary>日本語</summary>

ファイルは単一のオブジェクトです。各**キー**が画像パス、各**値**が `caption` / `tags` / `image_size` を任意で持つオブジェクトです。

注意点:

* `caption` / `tags` / `image_size` はいずれも任意です。これらをまったく持たないエントリも、（空キャプションの）有効な学習サンプルとして扱われます。
* `tags` は配列ではなく**文字列**です。指定された場合は §4 の方法でキャプションに連結されます。
* ファイルには最低 1 件のエントリが必要です。空のオブジェクトの場合、そのサブセットは警告とともにスキップされます。

</details>

## 3. JSON Lines format / JSON Lines 形式

Each line is an independent JSON object. The image path is given by the `image_path` key instead of being the object key.

```jsonl
{"image_path": "001.png", "caption": "a cat sitting on a sofa", "image_size": [1024, 768]}
{"image_path": "002.png", "caption": "a dog running on a beach", "width": 768, "height": 1024}
{"image_path": "subdir/003.png", "caption": "a bird", "tags": "bird, sky"}
```

Per-line fields:

| Key | Type | Required | Notes |
|---|---|---|---|
| `image_path` | string | **yes** | Image path (relative or absolute, see §6) |
| `caption` | string | no | Defaults to empty |
| `tags` | string | no | Merged into caption (see §4) |
| `image_size` | `[width, height]` | no | Image size |
| `width` / `height` | integer | no | Alternative to `image_size`; if both are present, `width`+`height` overrides `image_size` |

<details>
<summary>日本語</summary>

各行が独立した JSON オブジェクトです。画像パスはオブジェクトのキーではなく、`image_path` キーで与えます。

行ごとのフィールド:

| キー | 型 | 必須 | 備考 |
|---|---|---|---|
| `image_path` | 文字列 | **はい** | 画像パス（相対または絶対。§6 参照） |
| `caption` | 文字列 | 任意 | 省略時は空文字列 |
| `tags` | 文字列 | 任意 | キャプションに連結される（§4 参照） |
| `image_size` | `[幅, 高さ]` | 任意 | 画像サイズ |
| `width` / `height` | 整数 | 任意 | `image_size` の代替。両方が存在する場合は `width`+`height` が `image_size` を上書きする |

</details>

## 4. How `caption` and `tags` are combined / `caption` と `tags` の結合方法

The trainer combines `caption` and `tags` into the final training caption. The exact behavior depends on the subset's caption options (`caption_separator`, `enable_wildcard`), which are configured in the dataset config, not in the metadata file.

* **Default** (`enable_wildcard = false`): if `tags` is non-empty, it is appended to `caption` joined by `caption_separator` (default `,`):
  `final = caption + caption_separator + tags`. If `caption` is empty, `tags` is used alone.
* **Wildcard mode** (`enable_wildcard = true`): newlines in `tags` are first replaced by `caption_separator`; then `tags` is appended to **each non-empty line** of `caption`. This lets multi-line captions act as wildcard alternatives that all share the same tags.

If you do not need the distinction, you may simply put everything in `caption` and omit `tags`.

<details>
<summary>日本語</summary>

学習スクリプトは `caption` と `tags` を結合して最終的な学習キャプションを作ります。挙動はサブセットのキャプション関連オプション（`caption_separator`、`enable_wildcard`）に依存します。これらはメタデータファイルではなくデータセット設定で指定します。

* **デフォルト**（`enable_wildcard = false`）: `tags` が空でなければ、`caption_separator`（既定は `,`）で連結して `caption` の後ろに付与します。`final = caption + caption_separator + tags`。`caption` が空の場合は `tags` 単独になります。
* **ワイルドカードモード**（`enable_wildcard = true`）: まず `tags` 内の改行を `caption_separator` に置換し、その `tags` を `caption` の**空でない各行**に付与します。これにより、複数行キャプションを「同じタグを共有するワイルドカードの選択肢」として扱えます。

この区別が不要であれば、すべてを `caption` に入れて `tags` を省略しても構いません。

</details>

## 5. `image_size` / `image_size` について

`image_size` is `[width, height]` (width first). It is optional but **strongly recommended**, because it lets the trainer assign each image to an aspect-ratio bucket without opening the image file.

When `image_size` is absent, the trainer determines the size in this order:

1. From a matching latents cache file (`*.npz`) on disk, if latent caching is enabled and a cache file exists.
2. By reading the image file itself (slowest; only happens when needed, e.g. for resolution filtering).

If neither the image file nor a cache file is available, an entry with no `image_size` cannot be trained.

<details>
<summary>日本語</summary>

`image_size` は `[幅, 高さ]`（幅が先）です。任意ですが、**強く推奨**されます。これがあると、学習スクリプトは画像ファイルを開かずに各画像をアスペクト比バケツへ割り当てられるためです。

`image_size` がない場合、サイズは次の順序で決定されます。

1. latent キャッシュが有効で対応するキャッシュファイル（`*.npz`）がディスク上に存在すれば、そこから取得。
2. 画像ファイル自体を読み込んで取得（最も低速。解像度フィルタなど必要な場合にのみ実行）。

画像ファイルもキャッシュファイルもない場合、`image_size` を持たないエントリは学習できません。

</details>

## 6. Image path resolution / 画像パスの解決

Image paths (the object keys in JSON, or `image_path` in JSONL) may be relative or absolute:

* **Relative path**: resolved against the subset's `image_dir`. In this case `image_dir` is **required**; otherwise loading fails with an error.
* **Absolute path**: used as-is; `image_dir` is not needed.

Additional resolution behavior:

* If the path has no file extension, or the file does not exist at the resolved location, the trainer globs the directory for a file with the same base name and a supported image extension, and uses the first match.
* If no image file is found, a latents cache file (`*.npz`) sharing the base name is used instead (for both image size and training). This allows training purely from cached latents without the original images present.

<details>
<summary>日本語</summary>

画像パス（JSON ではオブジェクトのキー、JSONL では `image_path`）は相対・絶対のいずれでも構いません。

* **相対パス**: サブセットの `image_dir` を基準に解決されます。この場合 `image_dir` は**必須**で、無いとエラーで読み込みに失敗します。
* **絶対パス**: そのまま使用されます。`image_dir` は不要です。

追加の解決挙動:

* パスに拡張子が無い、または解決先にファイルが存在しない場合、学習スクリプトは同じベース名でサポート対象の画像拡張子を持つファイルをディレクトリ内から探索し、最初に一致したものを使用します。
* 画像ファイルが見つからない場合は、同じベース名を持つ latent キャッシュファイル（`*.npz`）が代わりに使用されます（画像サイズの取得と学習の両方に）。これにより、元画像が無くてもキャッシュ済み latent だけで学習できます。

</details>

## 7. Minimal valid examples / 最小の有効な例

Smallest possible standard JSON (captions only, sizes derived later):

```json
{
  "001.png": { "caption": "a cat" },
  "002.png": { "caption": "a dog" }
}
```

Smallest possible JSONL:

```jsonl
{"image_path": "001.png", "caption": "a cat"}
{"image_path": "002.png", "caption": "a dog"}
```

Both are valid. Adding `image_size` (and `tags` if desired) is recommended for performance and flexibility.

<details>
<summary>日本語</summary>

最小の標準 JSON（キャプションのみ。サイズは後から導出）:

最小の JSONL も上記のとおりです。

どちらも有効です。性能と柔軟性のため、`image_size`（必要なら `tags` も）を追加することを推奨します。

</details>
