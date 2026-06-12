# 20260612 P0特徴量・最小バックテスト実装（A1–A3）

## 目的

設計書 `results/20260612_roi_feature_recipes.md` のP0項目を実装する: (1) `field_size_running` バグ修正、(2) JSONL監査スクリプト、(3) レシピA1–A2（市場ミクロ構造）とA3（FLB較正ゾーンエンコーディング）、(4) 単勝フラット最小バックテスト、(5) リーク不変条件のテストスイート。

## 仮説

設計書 §2–§3 のとおり。本レポートは実装と検証のみで、実データでの仮説判定は未実施（理由は「使用データ期間」参照）。

## 変更内容

- `src/data/dataloader.py`
  - `field_size_running`（単勝が有効な行数 = 実出走頭数）を追加し `BASE_FEATURE_COLUMNS` に登録（X は37→38列）。
  - `popularity_pct` の分母を `field_size` から `field_size_running` に変更（取消馬混入バグの修正）。
  - `load_prepared_frame()` を追加（特徴量ビルダー/バックテスト用に、結果列込みの正規化フレームを返す。モデル直結禁止を docstring に明記）。
- `src/features/market.py`（新規・A1–A2): `race_overround`, `overround_excess`（前日までのexpanding中央値比）, `fav_odds`, `fav_dominance`, `gap_to_fav`, `log_odds_gap_prev/next`, `market_entropy`, `prob_x_entropy`, `pop_pct_x_dominance`。
- `src/features/trailing_stats.py`（新規・A3とその基盤):
  - `add_trailing_group_stats()`: 任意キーの日次集計 → **厳密に前日まで**の時間減衰累積（同日除外）を返す汎用エンコーダ。A4銘柄指数・C群EVマップもこの関数を再利用予定。
  - `add_odds_zone_calibration()`: オッズゾーン別の歴史的勝率（ゾーン自身の歴史的市場確率へ経験ベイズ縮約。履歴ゼロなら市場確率にフォールバック=自動見送り）、hit gap、ROI（全体ROIへ縮約）、有効サンプル数。
- `src/backtest/win_backtest.py`（新規): フラット100円単勝バックテスト。回収率・ROI・的中率・最大DD・年別/コース別/人気帯別分解。`ev_bet_mask()`（EV > 1+δ かつ n_eff 下限）。
- `scripts/audit_payout_cache.py`（新規): `_payout_cache.jsonl` のスキーマ無仮定監査（設計書§6.0）。キー/型/券種の棚卸し、raw CSVとのjoin率、サンプル構造ダンプを `results/` に出力。
- `scripts/run_p0_baseline.py`（新規): 実CSVでA1–A3→グリッド（δ×min_n_eff）→主要設定の分解評価→CLAUDE.md準拠レポートを自動保存。主要設定は事後選択を避けるため事前指定（δ=0.2, min_n_eff=1000, half_life=730日, k=200, burn-in=365日）。
- `tests/`（新規・17件): 後述。

## 使用データ期間

- **実データ未実行**。raw CSV/JSONLは `.gitignore` 対象でリモート実行環境に存在しない（設計書§1.1）。
- 検証は (a) 合成データのユニットテスト、(b) 合成市場（960レース・9,600行・控除20%・q∝p^0.85 の軽度FLB）でのスクリプトE2Eスモークで実施。

## 主要設定・ハイパーパラメータ

- オッズゾーン境界: [1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 100, ∞)
- 時間減衰半減期: 730日 / 経験ベイズ縮約 k: 200 / burn-in: 365日
- 購入: `calib_p_win_zone × 単勝 > 1+δ`、フラット100円、単勝は最終オッズ（購入・払戻とも最終オッズの自己完結評価）

## 評価指標（実装検証）

- `python3 -m pytest tests/ -q` → **17 passed**。内訳:
  - 同日レースが互いの履歴に入らない（同日除外）
  - 半減期1周期で履歴がちょうど0.5倍に減衰する
  - **未来・同日の結果を改変しても過去日の特徴量が不変**（リーク不変条件）
  - グループ間で履歴が混ざらない / 履歴ゼロ時は市場確率フォールバック / 縮約式の手計算一致
  - `field_size_running` が取消馬を除外し、`market_prob` が出走馬上で総和1
  - エントロピー・支配率・隣接ギャップ・overround超過（前日基準）の手計算一致
  - バックテストの投資/払戻/回収率/最大DD/分解の手計算一致、EVマスクの閾値・NaN挙動
- スモーク: 両スクリプトが exit 0 で完走し、レポートを自動保存。合成市場では主要設定（δ=0.2）が全見送り、δ=0で回収率0.85前後 — 控除20%・軽度歪みの合成市場では「ほぼ賭けない」が正しい挙動であり、見送りロジックが機能していることを確認。

## 検証コマンド

```bash
python3 -m pytest tests/ -q
python3 scripts/run_p0_baseline.py --results race_results.csv --meta race_meta.csv   # 要ローカル実データ
python3 scripts/audit_payout_cache.py --cache _payout_cache.jsonl                    # 要ローカル実データ
```

## 成功/失敗

実装・合成検証は成功。実データでの仮説判定は未実施（成功条件は設計書§6.1で事前固定済み）。

## 得られた知見

- pandas 3.0 ではcopy-on-writeにより `to_numpy()` が読み取り専用ビューを返すことがある。trailing encoder内の書き込みは `np.array(..., copy)` で明示コピーした（ローカルが pandas 2.x でも互換）。
- ゾーン較正の縮約先を「ゾーン自身の歴史的市場確率」にすると、履歴が薄いとき EV→1/overround < 1 となり自動的に見送りに倒れる。安全側のデフォルトとして妥当。
- 合成スモークで「歪みが小さい市場では賭けない」挙動を確認できた。実データでΔが出るかが次の判定点。

## Next Action

1. ユーザー（ローカル実データ環境）: 上記2コマンドを実行し、生成された `results/*_payout_cache_audit.md` と `results/*_p0_win_calibration_backtest.md` をコミットして共有。
2. Claude: P0実績を見て成功条件判定 → A4（銘柄過剰人気指数。`add_trailing_group_stats` のキー差し替えのみ）とB群（前走タイム差・上がり順位・通過パース）の実装仕様詳細化。
3. Codex: 下記レビュー観点の反証。

## Codexにレビューしてほしい点

- `add_trailing_group_stats` の同日除外・減衰・グループ境界の実装（tests/test_trailing_stats.py の不変条件で再現可能）。
- `popularity_pct` 分母変更の影響範囲（既存利用箇所はなし、X列数が37→38になる点の周知）。
- `run_p0_baseline.py` のグリッドが事後選択に使われない運用になっているか（主要設定の事前指定で担保したつもり）。
- バックテストの最大DDの定義（累積損益のピークアウト幅、ピーク下限0）で良いか。
