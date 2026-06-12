# 20260613 C4改訂版: O_fuku_min × P(top3) walk-forward

- 日付: 2026-06-13
- 目的: C4初版（P(place)×履歴平均複勝払戻EV）が全フォールド0賭けだったため、SPEC-2 `odds_final` の複勝レンジ下限 `O_fuku_min` を使う保守的EV判定へ改訂する。

## 実装

- 対象: `src/models/train_place_lgbm.py`
- 追加: `scraper/data/odds_final/*.jsonl` から `(race_id, horse_no) -> (fuku_min_odds, fuku_max_odds)` を読み込む `load_place_odds_range_map`
- EV判定: `EV = P̂(place) × fuku_min_odds`
- 決済: 従来どおり `_payout_cache.jsonl` の実複勝払戻 `place_payout`
- デフォルト主戦場: `--zone-lo 1.0 --zone-hi 1.4`
- 比較実行: `--zone-hi 2.0`

## 検証

```bash
py -m py_compile src/models/train_place_lgbm.py
py -m src.models.train_place_lgbm --smoke --features v3
py -m src.models.train_place_lgbm --features v3
py -m src.models.train_place_lgbm --features v3 --zone-hi 2.0
py -m pytest tests -q
```

- pytest: 36 passed
- C4改訂版の成果物:
  - `data/processed/place_lgbm_metrics_walkforward_fukumin_v3_z1_1p4.json`
  - `data/processed/place_lgbm_bets_walkforward_fukumin_v3_z1_1p4.csv`
  - `data/processed/place_lgbm_metrics_walkforward_fukumin_v3_z1_2.json`
  - `data/processed/place_lgbm_bets_walkforward_fukumin_v3_z1_2.csv`

## 結果: [1.0, 1.4)

| 年 | IG_val(zone) | zone行 | τ=0検証候補 | τ* | 賭数 | 全買いROI(zone) |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | +0.01964 | 74 | 1 | inf | 0 | 1.0000 |
| 2024 | +0.01509 | 79 | 5 | inf | 0 | 0.9620 |
| 2025 | +0.01127 | 94 | 9 | inf | 0 | 0.9840 |
| 2026 | +0.01388 | 41 | 12 | inf | 0 | 0.9707 |

## 結果: [1.0, 2.0)

| 年 | IG_val(zone) | zone行 | τ=0検証候補 | τ* | 賭数 | 全買いROI(zone) |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | +0.00475 | 768 | 2 | inf | 0 | 0.8729 |
| 2024 | +0.00106 | 579 | 5 | inf | 0 | 0.8946 |
| 2025 | +0.00190 | 848 | 10 | inf | 0 | 0.9052 |
| 2026 | +0.00532 | 306 | 14 | inf | 0 | 0.9291 |

## 判定

- `O_fuku_min` を使うと `[1.0,1.4)` のIGは初版より大きくなり、P(place)学習自体の余地は確認できた。
- ただし、EV>1候補は検証期間で最大12〜14件程度に留まり、`MIN_VAL_BETS=30` を満たさない。過剰適合を避ける資金投入ルールでは全フォールド `τ*=inf`、0賭けが正しい。
- 現時点の結論: **複勝本命帯には歪みがあるが、現行v3特徴量とSPEC-2最終複勝下限だけでは、再現可能な購入ルールにはまだ落ちない。**

## 次アクション

1. SPEC-1 `race_db` 払戻を正規化し、2015-2021を含めて `[1.0,1.4)` の12年安定性を確定する。
2. SPEC-2 `odds_final` 完了後に coverage を再監査し、C4を再実行する。
3. SPEC-3の時系列オッズが蓄積できたら、`O_fuku_min` を最終値ではなく購入時点レンジ下限に差し替える。
