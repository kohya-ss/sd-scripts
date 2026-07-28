# --avg_cp の挙動整理（sdxl_train_network.py）

## 対象コードと役割

- `train_network.py`  
  エポック終端で `--avg_cp` の判定・平均化・再ロード・optimizer 統計のリセットを行う。
- `library/avg_ckpt_util.py`  
  LoRA 重みだけを抽出し（`lora_` で始まるキーのみ）、平均（`uniform`/`ema`/`metric`）を計算する。
- `README.md`  
  オプション説明とフロー（raw 保存 → 平均 → 次エポック開始）が記載されている。

## 実装フロー（エポック終端の順番）

1. **raw ckpt の保存**（`--save_every_n_epochs` が有効な場合）  
   - `epoch+1 < num_train_epochs` のときのみ保存されるため、**最終エポックの raw は保存されない**。
2. **画像サンプルの生成**（`self.sample_images`）
3. **平均化 (`--avg_cp`)**
   - 条件: `(epoch + 1) / num_train_epochs >= avg_begin`
   - LoRA の `state_dict` を `cp_window`（最大 `avg_window`）に追加。
   - 窓が満杯になったら `average_state_dicts` で平均し、**平均後の重みをそのまま学習モデルにロード**。
   - `--avg_reset_stats` 指定時、Optimizer の `exp_avg` / `exp_avg_sq` / `exp_avg_max` をゼロ化（`step` は保持）。
     - 注: `AdamW8bit`（bitsandbytes）は optimizer state に `state1` / `state2` 等を使い、上記の `exp_avg` 系キーを持たないため、現行実装では実質何もリセットされない。
4. 次エポックは **平均後の重み** から開始。

### `--avg_mode ema` の重みのかかり方（窓サイズ N の例）

`alpha = 2 / (N + 1)` で、古い重みから順に EMA 更新する。  
`N=4` の場合 `alpha=0.4` となり、重み配分は以下の比率になる。

```
epoch_t-3 : 0.216
epoch_t-2 : 0.144
epoch_t-1 : 0.240
epoch_t   : 0.400
```

（古いほど重みが小さく、新しいほど大きい）

## 40エポック + 指定設定の具体例

前提:
- `max_train_epochs = 40`
- `avg_begin = 0.6`
- `avg_window = 4`
- `avg_mode = ema`
- `save_every_n_epochs = 1`

### 平均開始エポック

`(epoch + 1) / 40 >= 0.6` となる最初のエポックは **24**。  
よって **24 エポック目の終了時**から `cp_window` に溜まり始める。

### 具体的な平均対象

| エポック終端 | cp_window に入る | 平均の発動 | 平均対象 |
|---|---|---|---|
| 24 | 24 | しない | - |
| 25 | 24, 25 | しない | - |
| 26 | 24, 25, 26 | しない | - |
| 27 | 24, 25, 26, 27 | **する** | 24〜27 |
| 28 | 25, 26, 27, 28 | **する** | 25〜28 |
| 29 | 26, 27, 28, 29 | **する** | 26〜29 |
| … | … | … | … |
| 40 | 37, 38, 39, 40 | **する** | 37〜40 |

補足:
- 平均は **エポック終端で実行**されるため、**次のエポックは平均済み重み**から開始される。
- `save_every_n_epochs=1` の場合、raw ckpt は **1〜39 エポック**まで保存される。  
  40 エポック目の raw は保存されず、最終保存は **平均後の状態**で行われる。

## 「平均なし」を同時に作る方法はあるか

### 現状の挙動

- **raw ckpt は平均前に保存される**ため、同一エポックの「平均前スナップショット」は得られる。
- ただし平均が入った時点で学習モデル自体が変わるため、**「平均なしで最後まで学習した結果」とは一致しない**。

### 可能性のあるアプローチ

1. **別実行で比較する（最も正確）**  
   - `--avg_cp` あり／なしで 2 走させ、最終結果を比較。
2. **コード改修で“影の平均”を保存する**  
   - 平均を計算して別名で保存するが、学習モデルにはロードしない。  
   - もしくは平均後に保存してから元の重みへ戻す（学習挙動を変えない）。  
   - 現状そのためのフラグや分岐は **未実装**。
3. **学習後に後処理で平均する**  
   - 学習時は `--avg_cp` を使わず raw を保存。  
   - その後 `library/avg_ckpt_util.py` 相当で複数 ckpt を平均し、  
     「平均版」を別途生成する（学習中の平均化とは厳密には別）。

## まとめ

- `--avg_cp` は **LoRA 部分のみ**を対象に、学習の後半で窓平均を適用し、  
  **平均後の重みで学習を継続**する設計。
- 「平均あり」と「平均なし」を**完全に同時生成**する仕組みは現状ない。  
  比較には **別実行**が確実。  
  もしくは **平均版だけ別保存する改修**が必要。

## `--avg_cp_mode shadow` の追加仕様

### この機能の目的と位置づけ

この `shadow` は、いきなり「平均した重みを本採用して学習を続ける」のではなく、まず  
**平均候補が本当に raw より良いのかを、安全に観測するための段階**として追加された。

考え方は次のとおり:

- 学習本線はこれまでどおり `raw` のまま進める
- epoch 終端でだけ、`raw` と「直近 `avg_window` epoch を `avg_mode` で平均した center」を比較する
- 比較は backward を行わない軽い追加計算（forward-only scoring）で行う
- その結果を JSONL に記録し、
  - 平均候補が raw より勝つことが多いのか
  - margin を入れても勝つのか
  - 連勝する傾向があるのか
  を観測する

つまり、これは「avg_cp を自動採用する機能」の完成版ではなく、  
**平均を本採用する価値があるかを同一 run 内で検証するための観測モード**である。

将来的には、この観測結果が十分よさそうなら:

- score の良い center を条件付きで実際に学習モデルへ promote して次 epoch に進む

という拡張につなげるために、まず `shadow` で観測基盤を作り、その上に `promote` を追加した。  
`shadow` は **raw vs center を同条件で比較し、best を保存し、ログを取る** モード、`promote` はその比較結果を条件付きで次 epoch に反映するモードである。

### 目的

- **学習本線は raw のまま進める**
- 同一 run・同一 seed 条件で、`raw` と `current avg_mode で作る center 1本` を比較する
- VRAM 増加は常時ほぼゼロに寄せ、追加コストは epoch 終端の forward-only 採点時間と CPU RAM に寄せる

### CLI

- `--avg_cp_mode {live,shadow,promote}`
  - 既定値は `live`
  - `live` は従来どおり、平均後重みをそのまま学習モデルにロードする
  - `shadow` は平均候補を**採点と保存にだけ使い、学習モデルにはロードしない**
  - `promote` は平均候補を採点し、条件を満たしたときだけ次 epoch 開始点として center を採用する
- `--avg_promote_pick {fixed,best}`
  - 既定値は `fixed`
  - `promote` の `fixed` は `avg_mode` で指定した center だけを採点し、promote 判定に使う
  - `best` は proxy bank 上で `ema` と `uniform` を両方採点し、その epoch の良い方を promote 判定に使う
  - `shadow` はこの設定にかかわらず、観測用に `ema` と `uniform` を両方採点する
- `--avg_save_last_candidates`
  - 既定値は `False`
  - `shadow` / `promote` で学習終了時に `<output_name>_raw.safetensors` と `<output_name>_center.safetensors` を追加保存する
  - `best` でも `virtual_win_streak` は候補モードごとに分離せず 1 本で共有する。つまり前 epoch が `ema` 勝ち、次 epoch が `uniform` 勝ちでも連勝として数える
- `--avg_shadow_bank_size`
  - 既定値は `12`
  - train_proxy bank に保持する**学習バッチ数**
- `--avg_shadow_margin`
  - 既定値は `0.003`
  - `center_score < raw_score * (1 - margin)` のときだけ `center` 勝ち
- `--avg_shadow_patience`
  - 既定値は `2`
  - `shadow` では virtual streak のログ計算に使う
  - `promote` では center を本採用するための連勝条件にも使う
- `--avg_shadow_log_jsonl`
  - 既定値は `True`
  - `shadow` では `output_dir/avg_shadow+<output_name>.jsonl`、`promote` では `output_dir/avg_promote+<output_name>.jsonl` に epoch ごとの JSONL を出す

### 実装フロー

#### 1. 学習 step 中の train_proxy bank 収集

条件:

- `--avg_cp`
- `--avg_cp_mode shadow` または `--avg_cp_mode promote`
- `--train_batch_size 1`
- `(epoch + 1) / num_train_epochs >= avg_begin`
- final bank がまだ未確定

収集タイミング:

- `target` 算出後
- backward 前

この実装では、bank はいきなり先着順で固定しない。まず **construction pool** を集め、その中から最終的な fixed bank を選ぶ。

construction pool について:

- CLI は増やさず、内部仕様として自動サイズを使う
- target size は `max(avg_shadow_bank_size * 4, 32)`
- 例: `avg_shadow_bank_size=12` のとき、construction pool の target は `48`
- 現実装では **`train_batch_size=1` 専用**
  - `class_tokens` / `image_keys` の多様性判定を 1 サンプル単位で扱う前提のため
  - `avg_cp_mode=shadow/promote` で `train_batch_size!=1` はエラーにする

保持する情報:

- `noisy_latents`
- `target`
- `timesteps`
- `huber_c`
- `loss_weights`
- `input_ids`
- `input_ids2`（SDXL のとき）
- `network_multipliers`（存在するとき）
- `alpha_masks`（存在するとき）
- `original_sizes_hw`
- `crop_top_lefts`
- `target_sizes_hw`
- `text_encoder_outputs1_list`
- `text_encoder_outputs2_list`
- `text_encoder_pool2_list`
- `captions`（weighted captions 経路で必要なとき）
- `image_keys`
- `class_tokens`

実装上の保存方針:

- construction pool / final bank とも **CPU 保持のみ**
- `noisy_latents` / `target` / `alpha_masks` / cached TE 出力は CPU `fp16` に圧縮
- `timesteps` など整数テンソルはそのまま CPU に detach/clone
- `clean latents` は保持しない
- VAE 再実行もしない
- ノイズの再サンプリングもしない

#### 2. construction pool の開始 / 終了 / final bank 確定

開始タイミング:

- `avg_begin` に到達した最初の epoch の学習 step から開始する

終了タイミング:

- construction pool のサイズが内部 target に達した時点で、以後の候補収集を止める
- もし training end まで内部 target に達しなくても、`avg_shadow_bank_size` 以上の候補があれば最後に確定へ進む

final bank を決定するタイミング:

- epoch end で `avg_begin` 到達後
- `cp_window` 更新と同じタイミングで判定する
- construction pool が内部 target に達していれば、その epoch end で final bank を確定する
- internal target 未達でも最終 epoch なら、候補数が `avg_shadow_bank_size` 以上ある限り、その時点の construction pool から final bank を確定する

final bank の選び方:

- まず `class_tokens` の偏りが小さくなるように round-robin 的に拾う
- 同じ `class_tokens` の中では `image_key` が重ならない候補を優先する
- `class_tokens` が存在しないデータ形式では `image_key` ベースで分散させる
- final bank が一度確定したら **freeze** し、その後は最後まで同じ bank を使い続ける

この設計の意図:

- 少数画像・高 repeat のデータでは、先着順 freeze だと偏りやすい
- しかし epoch ごとに評価 bank を変えると、score 比較の意味が崩れる
- そのため「確定前だけ多様性を見る construction pool」と「確定後は固定の final bank」を分けている

#### 3. epoch end の raw snapshot / center 作成

epoch 終端の順番は以下:

1. 通常の `save_every_n_epochs` による raw ckpt 保存
2. `sample_images`
3. `clean_memory_on_device()` で一時メモリ解放
4. `avg_begin` 到達後なら、現在の LoRA state を `raw_sd` として CPU に抽出
5. `cp_window` と `cp_window_epochs` に追加

ここでの分岐:

- `avg_cp_mode=live`
  - 従来どおり `average_state_dicts()` で平均し、**平均後重みを学習モデルにロード**
  - `avg_reset_stats` も従来どおり有効
    - 注: `AdamW8bit` では現行のリセット対象キーが optimizer state に存在しないため、`--avg_reset_stats` / `--no-avg_reset_stats` の差は実質ない
- `avg_cp_mode=shadow`
  - **学習モデルは raw のまま**
  - `cp_window` が `avg_window` に満ち、かつ final bank 完成後のみ採点する
- `avg_cp_mode=promote`
  - まず `shadow` と同じ手順で採点する
  - `margin` と `patience` を満たしたときだけ center を次 epoch 開始点として採用する

#### 4. proxy scoring

shadow の採点は epoch 終端で行う。

比較対象:

- `raw_sd`
- `avg_mode` で `cp_window` から作った `center_sd`

採点手順:

1. 現在の raw モデルで `raw_score` を計算
2. RNG state を元に戻す
3. `center_sd` をモデルに一時ロード
4. 同じ bank で `center_score` を計算
5. 採点後に **必ず raw に戻す**
6. RNG state も元に戻し、shadow scoring が次 epoch の乱数系列を汚さないようにする

VRAM 方針:

- raw と center を GPU に同時常駐させない
- bank item は 1 件ずつ GPU に送る
- backward なし
- optimizer 更新なし
- `torch.no_grad()` で forward-only 採点

#### 5. score の定義

score は `train_proxy bank` 上の **mean loss**。

実装では、学習 step と shadow scoring の両方が同じ helper を使う:

- `NetworkTrainer._get_text_conds_for_batch()`
- `NetworkTrainer._compute_batch_loss()`

この helper 内で再利用している処理:

- `call_unet()`
- `train_util.conditional_loss()`
- `apply_masked_loss()`
- `loss_weights` 乗算
- `apply_snr_weight()`
- `scale_v_prediction_loss_like_noise_prediction()`
- `add_v_prediction_like_loss()`
- `apply_debiased_estimation()`

つまり、judge の loss 式は**通常学習の loss と同じ系統**を使っている。

補足:

- SDXL では `sdxl_train_network.py` の `get_text_cond()` が outer の grad mode を尊重するように修正されており、shadow scoring 中に不要な graph を強制生成しない

#### 6. winner 判定

- `center_score < raw_score * (1 - avg_shadow_margin)` のときだけ `winner = "center"`
- それ以外は `winner = "raw"`

あわせて以下を epoch ごとに計算・記録する:

- `virtual_margin_ok`
- `virtual_win_streak`
- `virtual_would_promote = (virtual_win_streak >= avg_shadow_patience)`

モードごとの差:

- `shadow`
  - これらは観測とログ専用
  - 学習モデルを center に切り替えることはしない
- `promote`
  - `winner = "center"` かつ `virtual_would_promote = true` のとき、center を次 epoch 開始点として採用する

#### 7. 保存物

通常の最終保存に加えて、必要なら以下を追加保存できる:

- `<output_name>_raw.safetensors`
- `<output_name>_center.safetensors`

条件:

- `--avg_cp_mode shadow` または `--avg_cp_mode promote`
- `--avg_save_last_candidates`

定義:

- `_raw`: 最終 epoch の学習終了時点の raw endpoint
- `_center`: 最終 epoch の `avg_mode` で作った center

保存実装:

- 現在の in-memory / CPU 上の LoRA state dict から直接 `.safetensors` を保存
- `save_every_n_epochs` に依存しない
- LoRA 部分だけを保存する
- SAI metadata と safetensors hash metadata も付与する

#### 8. JSONL logging

shadow 有効時は epoch ごとに 1 行の JSONL を出す。

主な出力項目:

- `epoch`
- `progress`
- `construction_pool_size`
- `construction_pool_target_size`
- `bank_ready`
- `bank_size`
- `bank_unique_class_tokens`
- `bank_max_per_class_tokens`
- `bank_unique_image_keys`
- `bank_max_per_image_key`
- `bank_timestep_bins`
- `avg_mode`
- `avg_promote_pick`
- `avg_window`
- `raw_proxy_loss`
- `center_proxy_loss`
- `ema_proxy_loss`
- `uniform_proxy_loss`
- `best_candidate_mode`
- `best_candidate_proxy_loss`
- `selected_candidate_mode`
- `selected_proxy_loss`
- `delta_abs`
- `delta_pct`
- `winner`
- `winner_mode`
- `virtual_margin_ok`
- `virtual_win_streak`
- `virtual_would_promote`
- `window_epochs`
- `status`

`status` の例:

- `before_avg_begin`
- `building_pool`
- `waiting_window`
- `scored`

各項目の意味:

- `construction_pool_size`
  - 現時点までに集まっている construction pool の候補数
- `construction_pool_target_size`
  - 内部仕様で決まる construction pool の目標候補数
- `bank_ready`
  - final bank が確定済みかどうか
- `bank_size`
  - final bank に実際に入っている候補数
- `bank_unique_class_tokens`
  - final bank に含まれる `class_tokens` の種類数
- `bank_max_per_class_tokens`
  - 1つの `class_tokens` が final bank 内で最大何枠を占めたか
- `bank_unique_image_keys`
  - final bank に含まれる `image_key` の種類数
- `bank_max_per_image_key`
  - 1つの `image_key` が final bank 内で最大何枠を占めたか
- `bank_timestep_bins`
  - final bank に含まれる timestep を `low` / `mid` / `high` の粗い3ビンに分けた分布
  - timestep の偏りが極端でないかを見るための指標

`image_keys` の扱い:

- `image_key` は batch 内部情報として常時持つ
- これは shadow 以外を直接変えるためではなく、将来の bank 解析やデバッグにも使い回せる軽い識別情報として載せている
- VRAM コストはほぼなく、CPU 側の文字列リストとして扱う

#### 9. 実際の制約

- 今回の実装で比較する center は **現在の `args.avg_mode` で作る 1 本だけ**
- `val_bank` は未実装
- `adaptive promote` は未実装
- resume 時は `cp_window` を既存 epoch ckpt から復元するが、**construction pool / final bank / best_* / streak は復元しない**
- `avg_cp_mode=shadow` / `avg_cp_mode=promote` は **`train_batch_size=1` のみ対応**
- 分散学習では score/save/log は main process で行う
  - ただし bank の all-gather はしていないため、**proxy bank は main process の local shard ベース**

### まとめ

- `shadow` は「学習は raw のまま」「平均候補は epoch end に観測だけ」のモード
- 判定 loss は学習と同じ helper 経路に寄せている
- 常時増える VRAM はほぼなく、追加コストは CPU bank と epoch end forward に寄せている
- 必要なら最終 `raw` / `center` を **1 run から取得できる**

## `avg_cp_mode=promote` の仕様

### 目的

`shadow` で観測していた `raw vs center` 比較を、条件付きで実際の学習継続に反映できるようにする。

初版では複雑化を避け、次の方針に限定する。

- promote 判定に実際に使う候補は `avg_promote_pick` で決める
  - `promote` の `fixed`: `current args.avg_mode` で作る center 1 本だけを作成・採点する
  - `best`: `ema` と `uniform` のその epoch の良い方
- `shadow` は従来どおり、観測用の `ema` と `uniform` を両方作成・採点する
- promote 判定は fixed proxy bank 上の score に基づく
- center が継続的に優勢と判断されたときだけ promote する
- promote 後も averaging 用履歴には **raw endpoint** を使う
- val_bank、adaptive rollback は初版では入れない

### 推奨初期設定

- `avg_window=4`
- `avg_begin=0.6`
- proxy bank は既存の diversity-aware fixed bank を使う

### 基本フロー

各 epoch で区別する状態は 2 つある。

- `raw endpoint`
  - その epoch の実学習終了時点の LoRA state
- `adopted model`
  - 次 epoch の開始点として採用する state
  - promote しなければ raw
  - promote すれば center

epoch end の流れ:

1. その epoch の学習終了時点を `raw endpoint` として CPU に保持する
2. `raw endpoint` を `cp_window` に追加する
3. `cp_window` が `avg_window` 個揃っていれば、`promote + fixed` は `avg_mode` のみ、それ以外は `ema` と `uniform` を作る
4. `best` かつ `avg_mode` が別モードなら、その center も作る
5. fixed proxy bank 上で `raw_score` と必要な候補 score を比較する
6. `avg_promote_pick` に応じて、promote 判定に使う候補を 1 本選ぶ
7. promote 条件を満たさなければ、次 epoch は raw で継続する
8. promote 条件を満たせば、次 epoch は選ばれた center をロードして開始する

重要:

- 学習継続には promote 済み center を使う
- ただし averaging 用履歴に積むのは、毎 epoch の **raw endpoint**
- promote 済み center 自身を履歴に直接積み続けない

この設計により、平均済みモデルの再平均に偏りすぎるのを防ぎつつ、各 epoch の実際の着弾点から center を推定し続ける。

### promote 条件

初版では次の条件をすべて満たしたときだけ promote する。

- `selected_score < raw_score * (1 - avg_shadow_margin)`
- `virtual_win_streak >= avg_shadow_patience`

ここで:

- `avg_shadow_margin`
  - center がわずかに良いだけでは promote しないための安全弁
- `avg_shadow_patience`
  - 1 回勝っただけで採用せず、連続優勢を要求する安全弁
  - ただし `avg_promote_pick=best` では「同じ候補が連勝したか」までは見ず、「その epoch に選ばれた候補が連続で raw に勝ったか」を見る簡略仕様とする

実装では `shadow` と同じ margin / patience をそのまま流用する。

### optimizer state

初版では、promote 時に `avg_reset_stats` 相当を使うことを推奨する。
ただし `AdamW8bit` では現行のリセット対象キーが optimizer state に存在しないため、このリセットは実質何もしない。

理由:

- promote は model weights の離散的な置き換えに近い
- 古い optimizer moments をそのまま引きずると不安定になりやすい

実装:

- `avg_cp_mode=promote` で promote が実行されたときのみ
  - `args.avg_reset_stats=True` なら optimizer state をリセットする
    - 注: `AdamW8bit` の `state1` / `state2` 等は現行実装ではリセット対象外
- promote が起きなかった epoch では何もしない

### score

score 定義は shadow と同じ。

- fixed proxy bank 上の mean loss
- lower is better

使用コードパスも shadow と同じにする。

- `NetworkTrainer._get_text_conds_for_batch()`
- `NetworkTrainer._compute_batch_loss()`

### ログ

JSONL は shadow と同じ形式をベースにしつつ、promote 用に以下の項目を追加する。

- `promote_mode`
  - promote 評価モードかどうか
- `promote_decision`
  - `"raw"` または採用候補モード名
- `promote_applied`
  - その epoch end で実際に center を採用したか
- `next_epoch_source`
  - 次 epoch が `raw` 起点か `center` 起点か
- `optimizer_reset_applied`
  - promote 時に optimizer reset を行ったか
- `ema_proxy_loss`
  - `ema` で作った center の proxy loss。`promote + fixed` で別モードを指定した場合は `null`
- `uniform_proxy_loss`
  - `uniform` で作った center の proxy loss。`promote + fixed` で別モードを指定した場合は `null`
- `best_candidate_mode`
  - `best` または `shadow` では `ema` と `uniform` のうち、その epoch の proxy loss が良かった候補。`promote + fixed` では `null`
- `selected_candidate_mode`
  - 実際に promote 判定へ使った候補

既存の

- `winner`
- `virtual_margin_ok`
- `virtual_win_streak`
- `virtual_would_promote`

はそのまま残してよい。

### 保存物

初版では通常の最終保存に加え、必要なら `--avg_save_last_candidates` で最終 `raw` / `center` を追加保存できる。

### 後方互換

- `avg_cp_mode=live` は既存挙動維持
- `avg_cp_mode=shadow` は既存挙動維持
- `avg_cp_mode=promote` を使わない限り新しい promote 挙動は走らない

### 初版で入れないもの

- 複数 center 候補の同時比較
- `uniform/ema/metric` を同時に試して best を選ぶ機能
- `val_bank`
- promote 後に raw へ戻す rollback
- promote 済み center をそのまま履歴 window に積む挙動
- step 単位の全履歴近似 center
- sub-epoch snapshot averaging

### 将来拡張

今後の拡張候補:

1. `avg_begin` と `promote_begin` の分離
2. step EMA center
3. `val_bank`
4. 複数候補比較

### `live` との違い

`avg_cp_mode=live` は、以前からある元祖の `avg_cp` 仕様であり、**promote で center を選び続けるのと同じではない**。

主な違い:

- `live`
  - `avg_window` 到達後は毎回 center を作り、**無条件でそのまま学習モデルへロードする**
  - `raw vs center` の score 比較をしない
  - margin / patience を見ない
  - その時点で学習モデルそのものが平均済み状態に置き換わる

- `promote`
  - 毎 epoch まず `raw endpoint` を保持し、それを履歴 `cp_window` に積む
  - raw と center を proxy score で比較する
  - margin と patience を満たしたときだけ center を採用する
  - 学習継続には center を使っても、履歴には raw endpoint を使う

つまり `live` は「常時平均を本線へ反映するモード」、`promote` は「score 条件を満たしたときだけ center を本採用するモード」であり、学習軌道も履歴の意味も異なる。
