# 20260612 v2特徴量強化（レシピA1–A4/B群）とIGゲート・早期停止のハーネス統合

## 目的

walk-forward実験v1の診断（`results/20260612_value_lgbm_walkforward.md`: プールROI 0.798、**IG<0 = 現特徴量では市場較正を超えられない**）を受け、同レポート Next Action #2（検証CE早期停止＋IGゲート運用）と #3（特徴量強化）を実装する。特徴量は設計書 `results/20260612_roi_feature_recipes.md` のレシピ群A1–A4・Bに対応。

## 仮説

市場が見ない次元（前走の時計・展開・トリップ・相手関係、銘柄の過大評価、ゾーン較正履歴）を shift(1)/前日まで集計で注入すれば、検証IG = CE市場(val) − CEモデル(val) > 0 に到達し、初めて賭け評価に意味が生じる。

## 変更内容

### 新規特徴量モジュール（計40列）

- `src/features/past_run.py`（B群・20列）: 勝ち馬タイム差（last/best/直近3平均）、スピード残差（コース×距離帯の**前日まで**expanding中央値基準; last/best/トレンド）、前走上がり順位pct・上がりtop2、通過パース（前走の序盤/終盤位置・ポジション上げ幅）、前走馬番pct、言い訳フラグ（closer_excuse=上がり最速級×後方×4着以下、excusable_loss_draw=大外×6着以下、flattered_win_inside=最内×連対）、前走レース強度（top3のmarket_prob平均）、オッズ/人気ドリフト、excuse_score、**overreaction_index = pop_drift × excuse_score**
- `src/features/brand.py`（A4・6列）: 騎手/調教師の overbet = E_hist[market_prob − is_win]（0方向へ縮約）、ROI（全体ROIへ縮約）、有効サンプル数。`add_trailing_group_stats` を再利用し同日除外を継承
- `src/data/dataloader.py`: B群の中間post-race列7個（time_behind_winner等）を `PROHIBITED_FEATURE_COLUMNS` に追加（防御強化）

### ハーネス統合（`src/models/train_value_lgbm.py`）

- `build_frame(feature_set)`: `--features v2` で `load_prepared_frame` 経由でA1–A4/B群を追加（v1パスは完全保存）。特徴量行列に `assert_no_feature_leakage` を実行
- **検証CE早期停止（デフォルトON）**: feval=勝者CE（市場オフセット込みsoftmax）、stopping_rounds=50、`metric="None"`。`predict_probs` は best_iteration を使用。v1の旧挙動再現は `--no-early-stop`
- **IGゲート（デフォルトON）**: IG_val = CE市場(val) − CEモデル(val) ≤ 0 のフォールドは τ*=∞（賭けゼロ）に強制。**ゲート判定はテストを覗かず検証期間のみで行う**。診断用 `--no-ig-gate`
- 結果テーブルに 木数・IG(val) 列を追加。成果物タグは v2 時 `*_v2`

## 使用データ期間

実データ未実行（raw CSVは本環境に不在）。検証は合成データ2種:

1. ユニットテスト（手計算ケース）
2. E2Eスモーク: 繰り返し出走する馬400頭×3年・騎手知名度バイアス（市場のみが反応する fame 項）を埋め込んだ合成市場1,232レース

## 主要設定・ハイパーパラメータ

v1から変更なし（lr=0.05, leaves=63, κ=0.1, β→8, λ→0.2, warmup 150/500）。追加: EARLY_STOP_ROUNDS=50。新特徴量の half_life=730日, shrinkage_k=200。

## 評価指標（実装検証）

- `python3 -m pytest tests/ -q` → **31 passed**（既存17 + 新規14）。新規分:
  - B群: 勝ち馬タイム差・上がり順位・通過パース・言い訳フラグ・レース強度・過剰反応指数の手計算一致 / **当該レース結果を全改変しても特徴量不変**（shift(1)起点の保証）/ スピード残差の基準が前日までであること / 中間post-race列が出力に漏れないこと
  - A4: overbet・縮約ROIの手計算一致 / 未来改変への不変性 / ヘルパー列の削除
- E2Eスモーク（合成市場・--smoke 80round）:
  - v1: 木数8で早期停止、IG(val)=+0.013、τ*=0.50、賭け0 → 旧来挙動の回帰OK
  - v2: 78特徴量で完走、木数6、IG(val)=+0.013、τ*=∞（検証対数富が全τで負 → 正しく全見送り）
  - 新40特徴量の非null率 ≥ 90%（欠損は初出走・本命の上隣不在など構造的なもののみ）

## 検証コマンド

```bash
python3 -m pytest tests/ -q
py -m src.models.train_value_lgbm --smoke --features v2     # 配管確認
py -m src.models.train_value_lgbm --features v2             # 本走（要ローカル実データ）
py -m src.models.train_value_lgbm --years 2025              # v1再走（早期停止の効果分離用）
```

## 成功/失敗

実装・統合・合成検証は成功。**IG>0仮説の実データ判定は未実施**（ユーザーのローカル実行待ち）。

## 得られた知見

- 早期停止だけでもv1の「+0.19 natsの過学習」はほぼ確実に解消される（合成でも木数500→6-8に縮small）。v2の効果は「早期停止後になおIG>0を積めるか」で測るべきなので、実データでは (a) v1+早期停止、(b) v2+早期停止 の両方を走らせ、IG差分を特徴量の寄与として分離すること
- `pd.to_numeric`（string dtype入力）はnullable Float64を返し `np.isnan` と非互換。特徴量モジュール側で float64 へ正規化した
- ハーネスのデフォルト挙動が変わった点に注意: 早期停止・IGゲートはデフォルトON（旧挙動は `--no-early-stop --no-ig-gate`）

## Next Action

1. ユーザー: ローカルで `py -m src.models.train_value_lgbm --features v2` と `--years 2025`（v1+早期停止）を実行し、IG(val)・CE・賭け結果を共有（メトリクスJSONは data/processed/ に保存される）
2. IG>0達成時: 賭け内訳のオッズ帯別分解で「どのレシピ群が効いたか」をLightGBM feature importanceとSHAP的寄与で確認 → 効かない群を剪定
3. IG≤0継続時: 設計書C群（払戻キャッシュEVマップ）へ前進（`scripts/audit_payout_cache.py` の実行が前提）
4. λ終端0.2→0.5の再実験（v1レポートNext Action #5）は特徴量判定後に実施

## Codexにレビューしてほしい点

- ハーネス変更がv1再現性を壊していないか（`--no-early-stop --no-ig-gate` でv1完全再現になるか）
- `make_val_ce_feval` のfeval仕様（LightGBM 4.xのcustom objective + fevalでpredsが生スコアで渡る前提）の確認
- B群 `speed_resid` の基準（コース種×距離帯200m×日次中央値のexpanding中央値）が荒すぎないか。競馬場・馬場状態欠落の影響範囲
- 早期停止の駆動指標を勝者CEにしたこと（全行binary loglossではなく）の妥当性
