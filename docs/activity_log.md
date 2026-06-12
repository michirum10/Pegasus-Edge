# Pegasus-Edge Activity Log

このファイルは、検証レポートとは別の共有作業台帳である。Codex と Claude は同じファイルに記録し、各エントリの `Actor` で担当を明示する。

詳細な実験・監査・バックテスト結果は `results/` に保存し、このログには「今まで何をやったか」「今何が進行中か」「次に何を見るべきか」を短く残す。

## Current Snapshot

- Updated: 2026-06-12 23:14 JST
- Current focus: v3特徴量（86個）のwalk-forward判定が完了。**IG_valは全6フォールドで非負化（+0.0000〜+0.0021 nat）したが経済的にはゼロ、運用は全年0賭け**。「現存4資産に対し単勝市場は効率的」の結論はレシピA1-A4/B群を加えても不変。次の主攻は (1) レシピ群C＝複勝EV検証（payout cache監査通過済み）、(2) SPEC-1/SPEC-3 外部データ蓄積。
- 確定事項: `_payout_cache.jsonl` スキーマ監査完了（8券種・14,054レース・2022-2026のみ・2024年に約1,100レース欠落、レシピ群C実装ゲート通過）。P0ゾーン単独戦略は失敗（非有界オッズゾーンのEVアーティファクトと診断）。pytest 31件は実環境で全通過。ブランチ `claude/confident-cerf-qumor0` は main へマージ済み（`55a1d3f`）。
- Active processes: スクレイピング用Pythonプロセスが2本稼働中の可能性あり。`race_db` と `odds_final` は件数が増える可能性がある。
- Latest confirmed root CSV range: `2015-01-04` から `2026-05-24`
- Latest `scraper/data` snapshot:
  - `race_db`: 8,224レース。2026完了、2025は1件不足、2024は残84件前後。
  - `odds_final`: 9,609レース。2026完了、2024完了、2025は1件不足、2023は取得中。
  - `odds_timeseries`: 2026-06-13分 36レース。`api_status=yoso` のため市場オッズではなく予想オッズ扱い。
- Next expected action: スクレイピング完了後に `scraper/data/*.jsonl` を正規化し、既存CSVとは別の拡張テーブルとして取り込む。

## Operating Rules

- 記録先はこの1ファイルに統一する。Codex用/Claude用に分けない。
- 各エントリには `Actor` を必ず書く。例: `Codex`, `Claude`, `User`, `Codex+Claude`。
- 検証レポートがある作業は `Report` に `results/*.md` をリンクする。
- コード・データ・スクレイピング・実験・監査・運用判断を行ったら、短く追記する。
- 詳細な数値や長い表は `results/` に置き、このログには要点と次アクションだけを書く。

## Log

### 2026-06-12 23:20 JST

- Actor: Codex
- Category: Code Review / Data Audit / Scraping
- Action: `results/` 配下の「Codexにレビューしてほしい点」を再スキャンし、SPEC-1/2/3、value LGBM、P0/v2/v3特徴量、ROI feature recipes、payout cache 関連の指摘を順番にレビュー。
- Findings: `src/data/extra_features.py` の `add_actor_history` は同日・同一レース内の過去行を含むため v2/v3 学習でリークする。SPEC-3 odds logger は取得開始前の遅延判定のみで、リトライ後に発走時刻を跨いだスナップショットを書ける。`value_objective.py` の勾配は有限差分で確認済み。`field_size_running`、trailing stats の同日除外、P0固定グリッド、max DD 定義、race_db/odds_final の主要パースは概ね妥当。
- Verification: `py -m src.models.value_objective` passed、`py -m pytest tests -q` は 31 passed。`scraper/data` はバックグラウンド取得中のため race_db/odds_final 件数は増加中。
- Status: Done
- Next: `add_actor_history` の同日・同一レース除外修正と、SPEC-3 odds logger の発走後書き込み防止を優先して実装する。

### 2026-06-12 23:14 JST

- Actor: Claude
- Category: Modeling / Backtest
- Action: v3特徴量（86個 = v2の45個 + レシピA1-A4/B群41個）のwalk-forward本走を完了し、IG>0判定を実施。
- Findings: IG_valは全6フォールドで非負（2021: +0.0021〜2022: +0.0000、平均+0.0006 nat）とv2（3フォールド負）から一貫して改善したが、経済的水準（+0.01 nat目安）には1桁届かず、全年τ*=∞で0賭け。**現存4資産の情報は単勝市場にほぼ完全に織り込まれている**ことがリポジトリ内の全特徴量ファミリーで確認された。未消化の可能性は券種変更（複勝・ワイド等の低流動性プール）と外部データのみ。
- Files touched: `src/models/train_value_lgbm.py`, `data/processed/value_lgbm_{bets,metrics}_walkforward_v3.{csv,json}`
- Report: `results/20260612_v3_feature_retest.md`
- Status: Done
- Next: レシピ群C（複勝EV・win-place tension）の実装。SPEC-1/SPEC-3蓄積継続。2024年payout cache欠落分の補完。

### 2026-06-12 23:06 JST

- Actor: Claude
- Category: Operations / Data Audit / Backtest
- Action: リモートブランチ `claude/confident-cerf-qumor0` を main にマージ（競合は `train_value_lgbm.py` のみ、main の二相学習版を採用）。ブランチ由来の特徴量群をハーネスに `--features v3` として接ぎ木し、実データで再テストを実施。
- Findings: (1) pytest 31件全通過（実環境）。(2) `_payout_cache.jsonl` 監査完了: ローマ字8券種キー・`{nums, yen}`形式・14,054レース・join率100%・ただし2022-2026のみで2024年に約1,100レース欠落 → レシピ群C（払戻EVマップ）は約4年分の履歴で実装可能。(3) P0ゾーン較正戦略は実データで失敗: 賭けの99%が10番人気以下に集中し回収率0.146。原因は非有界の最上位オッズゾーン内の異質性（ゾーン平均勝率×個別オッズのEVアーティファクト）。市場に無条件の正EVゾーンは存在しないという歪み監査の結論と整合。(4) v3スモークは IG_val=+0.0020 で完走。本走をバックグラウンド開始。
- Files touched: `src/models/train_value_lgbm.py`（v3接ぎ木）, `results/20260612_payout_cache_audit.md`, `results/20260612_p0_win_calibration_backtest.md`
- Report: `results/20260612_payout_cache_audit.md`, `results/20260612_p0_win_calibration_backtest.md`
- Status: Done（v3本走の結果は 23:14 のエントリ参照）
- Next: v3のIG_val 6フォールド分を確認し、IG>0達成可否で「特徴量寄与の分解」か「外部データ路線（SPEC-1/3）継続」かを判定する。

### 2026-06-12 22:43 JST

- Actor: Codex
- Category: Documentation / Operations
- Action: 共有活動ログの運用方針を決定。Codex と Claude で別ファイルにせず、`docs/activity_log.md` へ統一し、`Actor` 欄で担当を明示する方式を採用。
- Files touched: `docs/activity_log.md`, `AGENT.md`, `CLAUDE.md`
- Status: Done
- Next: 今後の作業ごとにこのログへ追記する。

### 2026-06-12 22:30 JST

- Actor: Codex
- Category: Data Audit
- Action: `scraper/data` を調査。JSONL、HTMLキャッシュ、ログ、稼働中Pythonプロセスを確認し、ルートCSV未反映の追加データがあることを確認。
- Findings: `race_db` は天候・馬場・発走時刻・horse_id/jockey_id/trainer_id・払戻を保持。`odds_final` は単勝・複勝レンジ・人気を保持。`odds_timeseries` は2026-06-13分36Rだが `api_status=yoso`。
- Status: Done
- Next: スクレイピング完了後、JSONLを正規化して拡張テーブル化する。

### 2026-06-12 22:10 JST

- Actor: Codex
- Category: Data Audit
- Action: ルート直下の `race_results.csv` / `race_meta.csv` の更新状況を再確認。
- Findings: 前回監査から行数・レース数・最新日付に差分なし。`race_results.csv` は526,813行・37,627レース、`race_meta.csv` は39,400レース、最新日は2026-05-24。
- Status: Done
- Next: スクレイピング成果物は `scraper/data` 側を確認する必要あり。

### 2026-06-12 19:18 JST

- Actor: Claude
- Category: Modeling / Feature Engineering
- Action: （リモートセッション・ブランチ `claude/confident-cerf-qumor0`）レシピB群（過去走ロールアップ20特徴量: 勝ち馬タイム差、前日まで基準のスピード残差、前走上がり3Fレース内順位、通過パース、言い訳フラグ、前走レース強度、過剰反応指数）とA4（騎手・調教師の overbet 指数 = 履歴上の市場確率−実勝率）を実装。ハーネスに検証CE早期停止とIGゲートを統合（mainの16:59と同方針の並行実装。マージ時はmain版採用）。
- Findings: 新規14テスト追加（手計算一致＋「当該レース結果を全改変しても特徴量不変」のリーク不変条件）。合成市場E2Eで配管確認。`pd.to_numeric`（string入力）のnullable Float64問題を特定しfloat64へ正規化。
- Files touched: `src/features/past_run.py`, `src/features/brand.py`, `src/data/dataloader.py`, `tests/test_past_run.py`, `tests/test_brand.py`
- Report: `results/20260612_v2_feature_reinforcement.md`
- Status: Done（実データ検証は23:06のv3再テストで実施）
- Next: 実データでIG再測定。

### 2026-06-12 17:18 JST

- Actor: Claude
- Category: Scraping
- Action: SPEC-1/SPEC-2スクレイパーを実装し、バックグラウンドでフルクロール開始。
- Findings: netkeibaオッズAPIで過去年も `status=result` と複勝レンジが取得できることを確認。複勝EV検証が射程に入った。
- Report: `results/20260612_spec1_spec2_scrapers.md`
- Status: In progress
- Next: クロール完了後、件数突合と欠損監査を行う。

### 2026-06-12 16:59 JST

- Actor: Claude
- Category: Modeling / Backtest
- Action: LightGBM v2としてIGゲートと特徴量強化を検証。
- Findings: 全フォールドで運用は0賭け。IGは実質ゼロで、現存特徴量だけでは市場を超える信号なし。ゲートは負ける戦略を止める挙動として成功。
- Report: `results/20260612_value_lgbm_v2_ig_gate.md`
- Status: Done
- Next: 真ID、天候、馬場、直前オッズなど外部情報を追加してIGを再測定する。

### 2026-06-12 13:56 JST

- Actor: Claude
- Category: Modeling / Backtest
- Action: 価値最大化LightGBM v1のwalk-forwardバックテストを実装・実験。
- Findings: プールROIは0.7979、95%CI上限が1.0未満でエッジなし。市場残差モデルは現特徴量では汎化せず、勝者の呪いを確認。
- Report: `results/20260612_value_lgbm_walkforward.md`
- Status: Done
- Next: IGゲート、早期停止、特徴量強化を入れる。

### 2026-06-12 13:50 JST

- Actor: Claude
- Category: Feature Engineering / Backtest
- Action: （リモートセッション・ブランチ `claude/confident-cerf-qumor0`）P0実装: `field_size_running` バグ修正（取消馬混入の頭数で `popularity_pct` を計算していた）、市場ミクロ構造特徴量A1-A2（overround超過・本命支配率・隣接logオッズギャップ・正規化エントロピー）、汎用trailingエンコーダ（日次集計→前日までの時間減衰累積＋経験ベイズ縮約）、FLBゾーン較正A3、単勝フラットバックテスト、`_payout_cache.jsonl` スキーマ無仮定監査スクリプトを実装。
- Findings: 同日リーク（騎手等の銘柄集計で同日他レースが混入する問題）を「前日まで」集計パターンで遮断する設計を確立。テスト17件で減衰・縮約・リーク不変条件を固定。
- Files touched: `src/features/market.py`, `src/features/trailing_stats.py`, `src/backtest/win_backtest.py`, `scripts/audit_payout_cache.py`, `scripts/run_p0_baseline.py`, `src/data/dataloader.py`, `tests/`
- Report: `results/20260612_p0_features_implementation.md`
- Status: Done
- Next: ローカル実データでの監査・バックテスト実行。

### 2026-06-12 13:40 JST

- Actor: Codex
- Category: Market Audit
- Action: 人気帯・オッズ帯別に全馬単勝100円フラット買いの素ROIを集計。
- Findings: 全体ROIは0.7301。長大穴帯は明確に低ROIで、favorite-longshot biasの存在を確認。
- Report: `results/20260612_market_distortion_audit.md`
- Status: Done
- Next: 条件付きの市場歪みを特徴量で探索する。

### 2026-06-12 13:35 JST

- Actor: Claude
- Category: Scraping / Odds Logger
- Action: SPEC-3時系列オッズロガーを実装・検証。
- Findings: 発売前は `api_status=yoso` としてnetkeiba予想オッズが返る。市場オッズとして扱えるのは `api_status=result` のみ。
- Report: `results/20260612_spec3_odds_logger.md`
- Status: Done
- Next: 開催日の朝にロガーを実走し、T-60からT-1の市場オッズ取得可否を確認する。

### 2026-06-12 13:29 JST

- Actor: Claude
- Category: Mathematical Design / Feature Engineering
- Action: （リモートセッション・ブランチ `claude/confident-cerf-qumor0`）現存4資産のみを材料とするROI直結特徴量レシピ集を設計。レシピ群A（大衆認知バイアス: FLB較正TE・ブランド過剰人気・市場ミクロ構造）、B（過去走ロールアップ: 言い訳可能な敗戦の抽出）、C（払戻キャッシュEVマップ: ゾーン×文脈TE・Harville較差・複勝テンション）の3群と、市場オフセット付き条件付きロジットによる「市場残差学習」原則を定義。
- Findings: データ実体はGitHubリポジトリに不在（gitignore対象）であることを確認し、スキーマ契約と監査値のみから設計。`field_size` への取消馬混入と銘柄集計の同日リークを設計段階で特定。
- Report: `results/20260612_roi_feature_recipes.md`
- Status: Done
- Next: P0（A1-A3＋最小バックテスト）の実装。

### 2026-06-12 13:19 JST

- Actor: Claude
- Category: Mathematical Design
- Action: 期待値直接最大化カスタム損失関数とバックテストプロトコルを設計。
- Findings: 市場残差パラメータ化、Soft-Kelly、CE較正アンカー、IGゲート、walk-forward検証規約を定義。複勝と時系列オッズは追加取得が必要と整理。
- Report: `results/20260612_value_loss_design.md`
- Status: Done
- Next: Codex側でリーク安全な実装・監査・反証を行う。

### 2026-06-12 12:49 JST

- Actor: Codex
- Category: Data Loader / Leak Audit
- Action: `race_results.csv` と `race_meta.csv` を監査し、リーク安全データローダーを実装。
- Findings: `race_id` many-to-one結合は成立。Xは522,548行・37特徴量、yは5列、禁止列混入なし。現CSVには複勝オッズ・公式払戻金・競馬場・馬場状態・天候・真IDが不足。
- Report: `results/20260612_data_loader_audit.md`
- Status: Done
- Next: 不足データの取得と単勝最小バックテストを進める。
