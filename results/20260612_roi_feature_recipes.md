# 20260612 ROI直結・特徴量エンジニアリング設計書（市場の歪みハック・レシピ集）

## 目的

外部スクレイピングに一切頼らず、現リポジトリのデータ契約（`race_results.csv` + `race_meta.csv` + ローカルの `_payout_cache.jsonl`）だけを材料に、最終オッズに織り込まれた大衆心理の系統的歪みを数値化し、回収率（ROI）を直接最大化するための特徴量レシピを、Codex が実装可能な精度で仕様化する。

本書は設計書である。モデル学習・バックテストは未実施（理由は §1.1）。

## 仮説（大枠）

1. パリミュチュエル市場の最終オッズは「平均的にはよく較正されているが、ゾーン・文脈・銘柄（騎手/調教師/馬主）・直近着順への過剰反応において系統的に歪む」。
2. その歪みは現有データ（オッズ・人気・過去走・払戻キャッシュ）だけで、リークなしにインデックス化できる。
3. モデルは市場確率を「打ち負かす」のではなく、市場確率をオフセットに固定し**残差（歪み）だけ**を学習するのが最もサンプル効率がよい。

---

## 1. 現存データ構造の自己インスペクション結果

### 1.1 リポジトリ実態（本セッションで検証済み）

- GitHub `michirum10/Pegasus-Edge` はブランチ `main` のみ、コミットは `e5b0174`（feat: add leak-safe data loader and agent charters）1件。
- ワークツリーおよびコンテナ全域を検索した結果、`race_results.csv` / `race_meta.csv` / `_payout_cache.jsonl` の**実体ファイルは存在しない**。`.gitignore` で明示的に除外されており（25行目に `_payout_cache.jsonl`）、raw データはユーザーのローカル環境にのみ存在する。
- したがって本書の「データ事実」は、(a) `src/data/dataloader.py` のスキーマ契約、(b) `results/20260612_data_loader_audit.md` の実測監査値、の2つの一次資料から復元した。実セル値・分布・`_payout_cache.jsonl` の正確なキー構造は本環境では検証不能であり、§6.0 の監査スクリプトをローカルで実行して確定させること。

### 1.2 race_results.csv（監査済み事実）

| 項目 | 値 |
|---|---|
| 行数 | 526,813（馬×レース） |
| 列数 | 18 |
| `race_id` ユニーク | 37,627 |
| 平均頭数 | ≈ 14.0（526,813 / 37,627） |
| 期間 | 2015-01-04 〜 2026-05-24（開催日 1,234日） |
| `着順` 数値化率 | 99.187%（非数値 4,284行: 中2,369 / 除1,066 / 取829 ほか降着・失格） |
| `単勝` / `人気` coverage | 99.640%（≒出走取消・除外馬がNaN） |
| `馬体重` parse coverage | 99.826% |

列: `race_id, 着順, 枠番, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, 通過, 上り, 単勝, 人気, 馬体重, 調教師, 馬主, 賞金(万円)`

### 1.3 race_meta.csv（監査済み事実）

39,400行 × 4列（`race_id, kaisai_date, course_type ∈ {turf, dirt, obstacle}, distance_m ∈ [1000, 4260]`）。`race_id` 重複0。results→meta の many-to-one left merge で未マッチ0。**meta側にのみ存在するレースが1,773件**ある（results未取得分。払戻キャッシュ結合監査時に突合せること）。

### 1.4 dataloader が既に生成する派生列（= 既存の武器）

- 基礎37特徴量: `frame_no, horse_no, assigned_weight, win_odds, popularity, course_type, distance_m, race_month, race_dayofweek, sex, age, horse_weight_kg, horse_weight_diff, jockey, trainer, owner, field_size, implied_win_prob, market_prob, popularity_pct, log_win_odds` + 馬名キーの過去走16特徴量（`shift(1)` 済み、win率・top3率・平均着順・前走オッズ/人気/距離/タイム/上がり・距離変化・中何日 等）。
- `y`: `finish_position, is_win, is_top3, win_payout_per_100yen, win_profit_per_100yen`（単勝×着順==1 による近似払戻）。

### 1.5 _payout_cache.jsonl（スキーマ未確定 — 契約案）

リポジトリ内にスキーマ定義・読込コードは存在しない。netkeiba 払戻テーブル由来と推定し、以下を**正規化契約案**とする（§6.0 の監査で確定するまで実装は仮置き禁止）:

```
1行 = 1レース想定: {"race_id": str, "単勝": [...], "複勝": [...], "枠連": [...],
  "馬連": [...], "ワイド": [...], "馬単": [...], "三連複": [...], "三連単": [...]}
正規化後 long 形式: (race_id, bet_type, combo: tuple[int,...], payout_per_100yen: int, combo_popularity: int|null)
```

### 1.6 構造上の注意点（リーク・バイアス検査で必ず見る箇所）

1. **`単勝` は最終（確定）オッズ**とみなす（netkeiba結果ページ由来）。発走前には観測不能な値であるため、バックテスト規約を「購入も払戻も最終オッズで自己完結」と固定する（パリミュチュエルでは払戻は最終プールから決まるので払戻側は正確。特徴量側は若干の情報優位を含む点をレポートに常記する）。
2. **`field_size` に出走取消馬が混入**: `prepare_model_frame` は `drop_invalid_market` の**前に** groupby size で頭数を計算している。`field_size_running = (単勝 notna) のレース内件数` を別途定義し、`market_prob`・`popularity_pct` の分母をこちらに揃えるべき（Codexレビュー依頼点）。
3. **同日リーク**: 馬単位の履歴は1日1走なので `shift(1)` で安全だが、騎手・調教師・ゾーン別の集計は同日に複数レースがあるため、**行シフトではなく「前日までの累積」**で計算する（§3.0 の日次cumsum-shiftパターンを必須とする）。
4. `馬名` キーは同名馬衝突リスクあり（暫定容認、`horse_id` 取得まで）。
5. `course_type == "obstacle"` は別市場（少頭数・特殊距離）。本レシピの対象から除外し、平地のみで学習・評価する。
6. `人気` は `単勝` の順位とほぼ冗長だが、同着オッズのタイブレーク情報を持つ。NaN（取消馬）はゾーン集計から除外。

---

## 2. 設計原則: 目的関数と「市場残差学習」

**目的関数**: レース内条件付きロジット（Plackett–Luce の勝者尤度）に市場確率を固定オフセットとして注入する。

```
P(馬iがレースrで勝つ) = softmax_i( βᵀ x_i + ln q_i ),   q_i = market_prob
損失 = −Σ_r ln P(勝ち馬_r)   （レース単位の多クラス交差エントロピー）
```

- β = 0 のとき モデル ≡ 市場。学習されるのは**市場からの乖離のみ**なので、本書の特徴量はすべて「市場が織り込み損ねる方向」を表すよう設計する。
- 購入条件: `EV_i = p̂_i × win_odds_i > 1 + δ`（δ: 安全マージン、初期値0.1〜0.2をグリッド）。レース内max EVが閾値未満なら**見送り**。
- 賭け金: 分数Kelly `f_i = c · (p̂_i·O_i − 1)/(O_i − 1)`、c = 0.1〜0.25。比較基準として均一100円賭けも常時併記（ROI・最大DD・賭け数）。

---

## 3. レシピ群A: 大衆の認知バイアス（過剰人気）を逆手に取る特徴量

### 3.0 共通実装パターン（リーク安全な expanding 統計）

すべての「歴史的較正・銘柄エンコーディング」は次の**日次cumsum-shift**で実装する（同日リーク遮断）:

```python
daily = df.groupby([KEY, "race_date"]).agg(n=("is_win","size"), w=("is_win","sum"),
                                           ret=("win_profit_per_100yen","sum"),
                                           q=("market_prob","sum"))
cum = daily.groupby(level=KEY).cumsum()
hist = cum.groupby(level=KEY).shift(1)          # ← 当日を除外した「前日まで」
df = df.merge(hist, on=[KEY, "race_date"], how="left")
```

時間減衰版は `decay = 0.5 ** (Δdays / 730)`（半減期2年）を日次集計に乗じてから累積する。少数サンプルは経験ベイズ縮約 `M̃ = (n·M + k·M₀)/(n + k)`（k≈200相当の擬似観測、M₀=全体平均）で潰す。

### A1. 市場確率と素朴確率の二重表現（基盤）

- **背景**: パリミュチュエルの控除率により `Σ 1/O ≈ 1/(1−t)`。`implied_win_prob = 1/O` は絶対価格、`market_prob = (1/O)/Σ(1/O)` は相対価格。両方持たせると「レース全体が荒れ含み（控除超過分の分布の歪み）」と「個別馬の相対評価」を分離できる。
- **計算**: 既存列。追加で `race_overround = Σ_race(1/win_odds)`（分母は `field_size_running`）、`overround_excess = race_overround − median_trailing(race_overround)`。

### A2. 本命–対抗ギャップ構造（クラウド合意の形状）

- **背景**: 人気順位は順序しか持たないが、大衆の「確信の形」はオッズの**間隔**に出る。1番人気1.5倍と1番人気3.2倍では同じrank=1でも全く別の市場。アンカリング（単勝1倍台への過剰集中）と中位帯の無関心（4〜8番人気の価格が雑）はFLB（favorite–longshot bias）の典型的発生源。
- **計算**:
  - `fav_odds = min_race(win_odds)`、`fav_dominance = q_(1)/q_(2)`（人気1位/2位のmarket_prob比）
  - `gap_to_fav = log_win_odds − log(fav_odds)`（全馬に付与）
  - `log_odds_gap_prev / next` = 人気順に並べた隣接馬とのlogオッズ差（「自分の上下が密集か孤立か」）
  - `market_entropy = −Σ q ln q / ln(field_size_running)`（0=一強、1=大混戦）
  - 交互作用: `market_prob × market_entropy`、`popularity_pct × fav_dominance`

### A3. 較正ギャップ・ターゲットエンコーディング（FLBの直接注入）

- **背景**: FLBは最も頑健な市場アノマリー: 大穴は過剰購入（回収率が大きく劣後）、堅い人気馬は控除率ほど損をしない。JRA公称払戻率は単勝80%だが、**ゾーン別の実効回収率は一様でない**。この「ゾーン別ROIマップ」を学習開始前に特徴量として注入すれば、モデルはFLBを再発見する必要がなくなる。
- **計算**（walk-forward必須）:
  1. `odds_zone = log_win_odds を分位ビン化（例: 16分位、または固定境界 [1.5,2,3,5,8,13,21,34,55,100]）`
  2. ゾーン×日次で §3.0 のcumsum-shift: `calib_roi_zone(z, t) = Σ_{過去} win_profit_per_100yen / (100·n)`（時間減衰+縮約）
  3. `calib_hit_gap(z, t) = Σ is_win/n − Σ market_prob/n`（実勝率−市場確率の歴史的乖離）
  4. さらに連続版: 過去データのみで `P(win) = σ(a + b·logit(market_prob))` をロジスティック再較正し、`calib_resid = ĝ(q) − q` を特徴量化。`b < 1` がFLBの符号（裾の過大評価）。
- **検証観点**: ゾーン境界を変えてもROI符号が安定すること。年別分解で2015–2019と2020–2026の符号一致を確認。

### A4. 騎手・調教師・馬主「ブランド過剰人気」指数

- **背景**: 有名騎手・有名厩舎・人気クラブ馬主の馬は、実力寄与以上に金が乗る（ハロー効果）。市場確率に対する**系統的な期待値残差**が銘柄ごとに持続するなら、それは大衆の値付けの癖そのものであり、fade（人気の裏）/follow（過小評価）の両方向に使える。
- **計算**（KEY = `jockey` / `trainer` / `owner` それぞれ、§3.0パターン）:
  - `jockey_overbet = Σ(market_prob − is_win) / n`（正 = このKEYの馬は市場確率が実勝率を上回りがち = 過剰人気）
  - `jockey_roi_resid = Σ(win_profit_per_100yen)/100n − calib_roi_zone(そのレースのodds_zone)`（ゾーン補正後の銘柄固有ROI）
  - 文脈分割版: KEY = `(jockey, course_type)`、`(jockey, distance_band)`、`(trainer, layoff_band)`（layoff_band = `horse_days_since_last_start` を [〜25, 26–60, 61–180, 181+] にビン化）。「休み明けに強い厩舎」を市場が織り込んでいない分が `(trainer, layoff_band)` のROI残差に出る。
- **リーク注意**: 必ず前日まで。乗り替わり初騎乗などサンプル0は縮約でM₀へ。

### A5. 人気順位×頭数の歴史的価格帯からの逸脱（zスコア）

- **背景**: 「18頭立ての3番人気は普段何倍か」には安定した経験分布がある。そこからの逸脱は、当該レース固有の情報か、群衆の過剰反応のどちらか。過去走情報（レシピ群B）と組み合わせると「説明のつかない逸脱 = 歪み」だけが残る。
- **計算**: KEY = `(popularity, field_size_band)` で `log_win_odds` の trailing 平均・分散を日次cumsum-shift（Σx, Σx², n）で持ち、`odds_rank_zscore = (log_win_odds − μ_hist)/σ_hist`。
- 補助: `pop_odds_tension = popularity_pct − market_prob のレース内rank pct差`（人気順位と確率質量の不整合検出）。

### A6. 構造的チート集（軽量・即実装可）

- `frame_no / horse_no` × `distance_band` × `course_type` のゾーンROIエンコーディング（内枠バイアスの市場織り込み残差。馬場状態がない分、`race_month` を交互作用に足して季節馬場を代理させる）
- `horse_weight_diff` 極端値フラグ（|Δ|≥10kg）× ゾーンROI（大幅増減への市場の過剰反応の符号を学習で決める）
- `age × popularity_pct`（高齢人気馬の過大評価/軽視の歴史的残差）

---

## 4. レシピ群B: 過去走ロールアップによる「言い訳可能な敗戦」の抽出

**背景となる大衆心理**: 群衆は直近着順に過剰反応する（recency bias / representativeness）。着順はトリップ・展開・距離適性・相手関係に汚染された極めてノイジーな順序統計量であり、「不利で大敗→次走で過剰に評価を下げられる」馬と「恵まれて好走→次走で過剰人気」の馬が系統的に発生する。現CSVの `タイム・着差・通過・上り` は**前走の値としてなら合法的に使える**未開拓資源である（現ローダーは `通過`・`着差`・レース内相対化を未使用）。

すべて `馬名` ごとに `race_date, race_id` ソート → `shift(1)`（またはshift(1)したrace_idで前走レース集計表をjoin）で実装する。

### B1. 勝ち馬からのタイム差（着順の脱ノイズ化・最重要）

- **計算**: 各過去レースで `winner_time = race_time_seconds where finish_position==1` をrace_id単位で作成 → 各馬の `time_behind_winner = own_time − winner_time` → `last_time_behind_winner = shift(1)`。
- **背景**: 8着0.6秒差（大接戦）と8着3.0秒差は市場の評価がほぼ同じ「8着」に圧縮される。前者は買い、後者は妥当。`着差` 文字列のパース（ハナ/クビ/アタマ/1/2…）はフォールバックとし、タイム差を主とする。
- 派生: `best_time_behind_winner_career`（キャリア最小値）、`avg_time_behind_winner_3`（直近3走移動平均）。

### B2. スピード指数lite（距離・コース内のタイム残差）

- **計算**: KEY = `(course_type, distance_m_band[200m刻み])` で `race_time_seconds` のtrailing中央値（日次cumsum-shiftの近似として、過去365日窓のローリング中央値を日次事前計算）→ `speed_resid = own_time − median_hist`。馬単位に `last_speed_resid`, `best_speed_resid_career`, `speed_resid_trend = last − mean(prev 2–4走)`。
- **背景**: 競馬場・馬場状態列がないため完全な指数は不可能だが、同一距離帯の同時期中央値で引くだけでも着順より遥かに連続的な能力量になる。市場は「着順」を見て、こちらは「時計」を見る、という非対称が狙い。
- **注意**: 馬場差を吸収しきれない点を明記。`race_month` 交互作用で季節分を部分吸収。

### B3. 上がり3F のレース内順位（差し届かず検出器）

- **計算**: 過去レース内で `last3f_rank_pct = rank(上り, ascending=True) / count_notna`。馬単位に `last_last3f_rank_pct`、`last3f_top2_flag = (rank ≤ 2)`。
- **背景**: 「上がり最速で4着以下」はスローペース展開の被害者の典型で、次走で人気を落とすが地力は割引不要のことが多い。日本の馬券市場で最も有名な歪みの一つだが、人気帯×頭数で条件付ければまだ残差が出る余地がある（市場が織り込んだ分は §2 のオフセットが吸収し、残差だけ学習される設計)。

### B4. 通過順パース（位置取りとトリップの言い訳）

- **計算**: `通過` "12-12-10-8" → `split('-')` → `early_pos_pct = first/field_size`, `late_pos_pct = last/field_size`, `ground_gained = early_pos_pct − late_pos_pct`。馬単位に `last_early_pos_pct`, `last_ground_gained`。
- **複合言い訳指数**: `closer_excuse = last3f_top2_flag × (last_early_pos_pct ≥ 0.7) × (last_finish_position ≥ 4)`（後方から上がり最速で届かなかった）。

### B5. 枠・馬番の不利フラグ（前走外枠大敗の割引回復）

- **計算**: `last_wide_draw = (last_馬番/last_field_size ≥ 0.75)`、`excusable_loss_draw = last_wide_draw × (last_finish_position ≥ 6)`。逆向き: `flattered_win_inside = (last_馬番/field ≤ 0.2) × (last_finish ≤ 2)`。
- **背景**: 枠の有利不利は市場も知っているが、「前走の枠不利を今走のオッズにどれだけ割り戻すか」は群衆が苦手な二段階推論であり、織り込み残差が出やすい。

### B6. 距離・コース替わりの言い訳と適性回帰

- **計算**: `horse_distance_change`（既存）に加え、`career_best_distance = speed_resid 最小の過去距離帯`、`dist_mismatch_last = |last_distance − career_best_distance|`、`back_to_best_dist = (今走距離帯 == career_best_distance band) × (last_distance band ≠ ...)`、`surface_switch_back =（前走がキャリア成績劣位のコース種別 → 今走が優位側）`。
- **背景**: 「距離延長で凡走→短縮で巻き返し」は市場の織り込みが鈍い古典パターン。前走凡走の理由を距離に帰属できる馬だけを拾う。

### B7. 前走レースの強さ（相手関係の言い訳）

- **計算**: 各レース終了後に `race_strength = mean(market_prob of 着順≤3 の馬)` と `race_top3_time_mean_resid`（B2残差のtop3平均）を確定 → 馬の前走race_idにjoin → `last_race_strength`。
- **背景**: 強敵相手の6着と弱メン相手の6着を市場は同列に扱いがち。`last_finish_position × last_race_strength` の交互作用が「価値ある敗戦」を拾う。

### B8. 市場ドリフトの過剰反応指数（本丸）

- **計算**:
  1. `odds_drift = log(win_odds_today) − log(horse_last_win_odds)`、`pop_drift = popularity_pct_today − last_popularity_pct`
  2. `excuse_score = w₁·closer_excuse + w₂·excusable_loss_draw + w₃·dist_mismatch_last_norm + w₄·(last_race_strength z) + w₅·(last_time_behind_winner が着順の割に小さい指標)`（初期wは全て1、後に条件付きロジットで自動学習）
  3. **`overreaction_index = pop_drift × excuse_score`**（人気を落とした×言い訳がある = 買いゾーン）、対称形 `flattered_index = −pop_drift × flatter_score`（人気を上げた×恵まれ要素 = 消しゾーン）
- **背景**: これは「群衆の更新則（着順アンカー）」と「正しい更新則（時計・展開・相手補正）」の差分を直接張る特徴量で、本書で最もROIに直結すると考える複合シグナル。

### B9. その他のロールアップ素材（優先度中）

- `prize_cum_before`（賞金累積=クラス代理。**過去走分のみ**、当該レース賞金は禁止列）/ `prize_per_start`
- `jockey_switch = (騎手 ≠ 前走騎手)` × 騎手ランク差（A4のovebet指数差で代理）— 乗り替わりの市場過剰反応
- `assigned_weight_delta = 斤量 − 前走斤量`
- `season_gap_pattern`: `horse_days_since_last_start` のビン × B2残差trend（鉄砲駆けの個馬性は `馬名×layoff_band` ではサンプル不足のため、調教師側 A4 で持つ）

---

## 5. レシピ群C: 払戻キャッシュ（JSONL）による期待値マップ注入

**背景**: 単勝プールは最も効率的で、複勝・馬連・三連系など組合せ系プールは (i) 参加者の計算力制約、(ii) 流し/BOX購入の構造的偏り（人気馬絡みの組合せに過剰質量）、(iii) プール規模が小さくノイズ大、により**単勝から導かれる公正価格（Harville近似）との乖離が持続**する。`_payout_cache.jsonl` の実払戻は、この乖離の歴史地図を作る唯一の材料である。

### C0. 前提（スキーマ確定後に着手）

§6.0 の監査スクリプトで bet_type ごとの存在率・join率・payout型を確定し、§1.5 のlong形式に正規化してから以降を実装する。`race_results` 側に存在しない1,773件のmeta-onlyレースとの突合も監査に含める。

### C1. 実効控除率と「公称とのズレ」推定

- **計算**: bet_typeごとに `effective_takeout(t) = 1 − Σ payout_hit / Σ total_pool想定`は不可（プール総額がない）ため、**的中組合せの平均回収率**で代理する: 全組合せ均等買い仮定の合成回収率 `R_synth(bet_type) = Σ payout / (100 × Σ n_combos)`（n_combos = field_size_runningから計算: 馬連 C(n,2)、馬単 P(n,2)、三連複 C(n,3)、三連単 P(n,3)、複勝 n）。
- **用途**: ゾーンROI（C2）のベースライン。公称払戻率（単勝・複勝80%、枠連/馬連/ワイド77.5%、馬単/三連複75%、三連単72.5%）との一致を健全性チェックとし、ズレたら正規化バグを疑う。

### C2. ゾーン×レース文脈のEVマップ（ターゲットエンコーディングの応用・本丸）

- **ゾーン定義**（組合せ→人気ランクパターンへの写像で次元圧縮）:
  - 単勝/複勝: 人気帯 z ∈ {1, 2–3, 4–6, 7–9, 10+}
  - 馬連/ワイド/馬単: ソート済み人気帯ペア（馬単は順序保持）例: (1,2–3), (1,4–6), (2–3,7–9)…
  - 三連複/三連単: ソート済み人気帯トリプルのクラス（例: {1,2–3,4–6}, {2–3,4–6,7–9}, {1,7–9,10+}…）
- **レース文脈シグネチャ**: `s = (course_type[2] × distance_band[4] × field_size_band[3] × market_entropy 三分位[3] × fav_odds帯[3])` ≈ 最大864セル。37,627レースに対し平均40レース/セル強。スパースセルは縮約で全体平均へ。
- **EVマップ**:
  ```
  M(s, bet_type, z, t) = Σ_{過去, decay} [zone内の的中払戻合計] / (100 × Σ_{過去, decay} [zone内の組合せ数])
  ```
  時間減衰半減期2年、経験ベイズ縮約 k≈200。**walk-forward必須**: 日次cumsum-shift（§3.0）をセル単位で適用。
- **特徴量注入**（馬行ベクトルへ）:
  - `ev_map_win_self = M̃(s, 単勝, 自分の人気帯)`、`ev_map_place_self = M̃(s, 複勝, 同)`
  - `ev_map_quinella_with_fav = M̃(s, 馬連, (自帯, 1番人気帯))`、`ev_map_trio_best = max over 自分を含むzoneクラスの M̃`
  - レースレベル: `ev_map_race_chaos = M̃(s, 三連複, 全部4番人気以下クラス)`（荒れ配当の歴史的うまみ）を全行にブロードキャスト
- **なぜ効くか**: モデルが「このオッズ構成・この文脈では歴史的にどのゾーンが過剰払戻だったか」を事前知識として持ち、勝率推定（§2）と独立に「払戻側の歪み」を係数1本で取り込める。

### C3. Harville公正価格との較差（組合せ系の構造的過小・過大人気の分離）

- **公正価格**: 勝率ベクトル p（= market_prob、後に較正後 p̂）から
  - 馬単 `P(i→j) = p_i p_j/(1−p_i)`、馬連 `P({i,j}) = p_i p_j [1/(1−p_i) + 1/(1−p_j)]`
  - 三連単 `P(i→j→k) = p_i · p_j/(1−p_i) · p_k/(1−p_i−p_j)`、三連複は6順列和
  - 複勝（top3入着）はHarville和、または減衰Harville（2着位置 p^0.8、3着位置 p^0.65 で再正規化）を採用
- **較差マップ**: `H_gap(s, bet_type, z) = M̃_observed − (1−takeout) / E_hist[Harville fair price of zone]`。観測ROIがHarville想定を上回るゾーン = 群衆が構造的に買わないゾーン（例: 中穴同士の組合せ）、下回るゾーン = 流し купなどで過剰質量（例: 1番人気絡み全般）。
- **用途**: 単勝モデルの p̂ を組合せ系に展開する際の購入ゾーンフィルタ。かつ `H_gap` 自体を馬行特徴量に落とす（自分が属する最有利ゾーンの較差値）。

### C4. 複勝テンション（同一馬への2つの市場価格の不整合）

- **計算**: 複勝払戻実績から `E[place_payout | win_odds帯, field_size帯]` の歴史マップを構築（cumsum-shift）。各馬に `expected_place_roi = M̃_place / (Harville-top3確率 × 想定払戻)` と、`win_place_tension = logit(較正top3確率) − logit(market_prob由来top3確率)` を付与。
- **背景**: 単勝と複勝は同一馬への独立プール。複勝側は「とりあえず人気馬の複勝」買いで歪みやすく、逆に堅実な中位人気馬の複勝が構造的に残ることがある。テンションが正に大きい馬は複勝期待値買いの第一候補で、**現データ契約で唯一実装可能な「第二の券種」のEV評価**。
- **注意**: 複勝払戻は同着・出走頭数（7頭以下は2着まで）で変則になるため、正規化時に頭数を必ず保持。

### C5. 払戻実績による y の置換（バックテスト精度向上）

特徴量ではないが必須改修: 現在の `win_payout_per_100yen = 単勝×100` 近似を、キャッシュの実払戻（単勝）で置換・突合する。差異があれば（同着・特払い等）実払戻を正とする。`y` に `place_payout_per_100yen` を追加し複勝バックテストを解禁する。**これらは y / バックテスト専用列であり X への投入は禁止**（`PROHIBITED_FEATURE_COLUMNS` に追加すること）。

---

## 6. 検証計画

### 6.0 STEP 0: ローカルで実行するデータ確定監査（Codex/ユーザー向け）

```python
# payout_cache_audit.py — スキーマ無仮定の監査。results/ にレポートを残すこと。
import json, collections, pandas as pd
keys = collections.Counter(); types = collections.Counter(); n = 0; race_ids = set()
with open("_payout_cache.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line); n += 1
        race_ids.add(str(rec.get("race_id", rec.get("id", ""))))
        for k, v in rec.items():
            keys[k] += 1; types[f"{k}:{type(v).__name__}"] += 1
print(n, "lines"); print(keys.most_common()); print(types.most_common(30))
meta = pd.read_csv("race_meta.csv", dtype={"race_id": str})
res = pd.read_csv("race_results.csv", dtype={"race_id": str}, low_memory=False)
print("join vs meta:", len(race_ids & set(meta.race_id)), "/", len(race_ids))
print("join vs results:", len(race_ids & set(res.race_id.unique())))
# 先頭3レコードの構造ダンプで bet_type 配下の形式（list/dict/文字列金額）を確定する
```

合格条件: 全行JSONパース可、`race_id` join率 ≥ 95%、8券種の存在率と payout の数値化率を確定。**この監査レポートが出るまでレシピ群Cの実装に着手しない。**

### 6.1 分割・評価プロトコル

- 学習: walk-forward（`iter_time_series_splits` 利用、年単位5フォールド目安。最終ホールドアウト: 2024-06-01以降は一切のマップ・較正の事前計算からも遮断）。
- 指標: ROI、回収率、的中率、賭け数、投資額、払戻額、最大ドローダウン（日付順の累積損益）、年別 / course_type別 / 距離帯別 / 人気帯別の分解。
- 統計的健全性: **レース単位クラスタ・ブートストラップ**でROIの95%CI（同一レース内の賭けは相関するため行ブートストラップ禁止）。
- 成功条件（事前固定）: ホールドアウトで (a) フラット賭けROI > 0.95 かつ ベースライン（全1番人気買い）+5pt以上、(b) 賭け数 ≥ 500、(c) 年別ROIの符号が2/3以上の年で一致、(d) 最大DDが総投資額の30%未満。
- 失敗条件: 上記未達、または特定年・特定人気帯の寄与が利益の70%超（偶然偏り）。失敗でもレポートを `results/` に保存。

### 6.2 リーク検査チェックリスト（Codexレビュー依頼）

1. すべての歴史マップ・銘柄エンコーディングが**前日まで**（同日除外）の集計か
2. `calib_*`, `ev_map_*`, `*_roi_resid` の事前計算がfold境界を跨いでいないか（ターゲットエンコーディング古典リーク）
3. B群の前走特徴量が `shift(1)` 起点か、前走race_id経由joinに未来行が混ざらないか
4. `field_size_running` 導入後の `market_prob` 再計算の整合
5. `PROHIBITED_FEATURE_COLUMNS` に `place_payout_per_100yen` ほか新規払戻列を追加済みか
6. obstacle除外・取消馬除外のタイミング一貫性

---

## 7. 実装優先順位（ROI期待値 × 実装コスト）

| 優先 | レシピ | 理由 |
|---|---|---|
| P0 | §6.0 JSONL監査 + C5 実払戻置換 | バックテストの土台精度 |
| P0 | A3 較正ギャップTE + A1/A2 | 最小実装でFLBを即収穫、以降の全評価の基準 |
| P1 | B1/B2/B3/B4 + B8 過剰反応指数 | 本書の中核エッジ。素材列が完全未開拓 |
| P1 | A4 銘柄過剰人気指数 | 実装は§3.0パターンの使い回し |
| P2 | C2 EVマップ + C4 複勝テンション | スキーマ確定後。第二券種の解禁 |
| P3 | A5/A6, B5–B9, C3 | 残差の積み増し |

## 8. 得られた知見（本セッション）

- リポジトリにデータ実体はなく、スキーマ契約と監査値のみが共有資産である。**レシピは全て監査済みの実在列だけを前提に設計した**（CLAUDE.md「存在しない列を前提にしない」遵守。`_payout_cache.jsonl` のみ契約案+監査ゲート方式とした）。
- 現ローダーは `着差・通過・レース内相対統計（上がり順位・タイム差）` を未使用であり、ここが最大の未開拓資源。
- `field_size` の取消馬混入と、銘柄系集計の同日リークが既知の落とし穴として要修正・要防御。

## Next Action

1. ユーザー: ローカルで §6.0 監査を実行し、出力を `results/20260612_payout_cache_audit.md` として共有。
2. Codex: `field_size_running` 修正、P0特徴量（A1–A3）と単勝最小バックテストの実装。
3. Claude: P0結果を受けて δ・縮約k・減衰半減期の感度分析設計、B群実装仕様の詳細化。

## Codexにレビューしてほしい点

- §6.2 チェックリスト全項目。特にターゲットエンコーディングのfold境界と同日リーク。
- C1 の合成回収率定義が「プール総額なし」の制約下で妥当か（より良い不偏推定があれば反証歓迎）。
- B2 スピード指数liteの基準窓（365日ローリング中央値）の実装コストと、月次キャッシュ化の提案。
