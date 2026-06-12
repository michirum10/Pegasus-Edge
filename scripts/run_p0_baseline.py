"""P0 baseline: favorite-longshot calibration strategy on win bets.

Implements report 20260612 recipes A1-A3 end to end on the real CSVs and
writes the experiment report required by CLAUDE.md §7.  The strategy is fully
online (trailing, strictly-before-date statistics), so every bet decision is
walk-forward by construction; yearly decomposition doubles as the
out-of-sample stability check.

Primary configuration is prespecified to avoid post-hoc picking:
delta=0.2, min_n_eff=1000, half_life=730d, shrinkage_k=200, burn_in=365d.

Usage (local machine, where the data lives):
    py scripts/run_p0_baseline.py --results race_results.csv --meta race_meta.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.win_backtest import WinBacktestResult, ev_bet_mask, run_flat_win_backtest
from src.data.dataloader import load_prepared_frame
from src.features.market import add_market_microstructure_features
from src.features.trailing_stats import add_odds_zone_calibration

DELTA_GRID = (0.0, 0.1, 0.2, 0.3)
MIN_N_EFF_GRID = (200.0, 1000.0)
PRIMARY = {"delta": 0.2, "min_n_eff": 1000.0}


def fmt_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "(no bets)"
    formatted = table.copy()
    formatted["recovery_rate"] = formatted["recovery_rate"].map(lambda v: f"{v:.3f}")
    formatted["hit_rate"] = formatted["hit_rate"].map(lambda v: f"{v:.3f}")
    return "```text\n" + formatted.to_string() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="race_results.csv")
    parser.add_argument("--meta", default="race_meta.csv")
    parser.add_argument("--half-life-days", type=float, default=730.0)
    parser.add_argument("--shrinkage-k", type=float, default=200.0)
    parser.add_argument("--burn-in-days", type=int, default=365)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frame = load_prepared_frame(args.results, args.meta, include_history=False)
    frame = frame.loc[
        frame["course_type"].isin(["turf", "dirt"])
        & frame["finish_position"].notna()
        & frame["win_odds"].notna()
        & frame["popularity"].notna()
    ].copy()

    frame = add_market_microstructure_features(frame)
    frame = add_odds_zone_calibration(
        frame, half_life_days=args.half_life_days, shrinkage_k=args.shrinkage_k
    )

    start_date = frame["race_date"].min()
    eval_start = start_date + pd.Timedelta(days=args.burn_in_days)
    eval_frame = frame.loc[frame["race_date"] >= eval_start].copy()

    all_mask = pd.Series(True, index=eval_frame.index)
    baseline_all = run_flat_win_backtest(eval_frame, all_mask)
    baseline_fav = run_flat_win_backtest(eval_frame, eval_frame["popularity"].eq(1))

    grid_rows = []
    primary_result: WinBacktestResult | None = None
    for delta in DELTA_GRID:
        for min_n_eff in MIN_N_EFF_GRID:
            mask = ev_bet_mask(eval_frame, delta=delta, min_n_eff=min_n_eff)
            result = run_flat_win_backtest(eval_frame, mask)
            grid_rows.append(
                {
                    "delta": delta,
                    "min_n_eff": min_n_eff,
                    "n_bets": result.n_bets,
                    "recovery_rate": round(result.recovery_rate, 4) if result.n_bets else float("nan"),
                    "hit_rate": round(result.hit_rate, 4) if result.n_bets else float("nan"),
                    "max_drawdown": round(result.max_drawdown, 0),
                }
            )
            if delta == PRIMARY["delta"] and min_n_eff == PRIMARY["min_n_eff"]:
                primary_result = result
    grid = pd.DataFrame(grid_rows)
    assert primary_result is not None

    today = dt.date.today()
    lines: list[str] = []
    lines.append(f"# {today:%Y%m%d} P0 win calibration backtest")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("レシピA1-A3（市場ミクロ構造 + FLB較正ゾーンエンコーディング）だけで、単勝フラット賭けの期待値ゾーンが実データに存在するかを確認する。")
    lines.append("")
    lines.append("## 仮説")
    lines.append("")
    lines.append("ゾーン別の歴史的勝率（経験ベイズ縮約・時間減衰・前日まで）× 最終オッズが 1+delta を超えるゾーンは、評価期間でも回収率がベースラインを上回る。")
    lines.append("")
    lines.append("## 使用データ期間")
    lines.append("")
    lines.append(f"- 全期間: {frame['race_date'].min():%Y-%m-%d} 〜 {frame['race_date'].max():%Y-%m-%d}")
    lines.append(f"- burn-in: {args.burn_in_days}日（評価は {eval_start:%Y-%m-%d} から。trailing統計はそれ以前も使用）")
    lines.append(f"- 対象: 平地（turf/dirt）、着順・単勝・人気が有効な行 = {len(frame):,}行 / 評価対象 {len(eval_frame):,}行")
    lines.append("")
    lines.append("## 主要設定・ハイパーパラメータ")
    lines.append("")
    lines.append(f"- half_life_days={args.half_life_days}, shrinkage_k={args.shrinkage_k}")
    lines.append(f"- オッズゾーン境界: [1,1.5,2,3,5,8,13,21,34,55,100,inf)")
    lines.append(f"- 事前指定の主要設定: delta={PRIMARY['delta']}, min_n_eff={PRIMARY['min_n_eff']}（グリッドは感度確認用）")
    lines.append("- 購入: EV = calib_p_win_zone × 単勝 > 1+delta、フラット100円")
    lines.append("- 単勝は最終オッズ（購入・払戻とも最終オッズで自己完結評価）")
    lines.append("")
    lines.append("## ベースライン")
    lines.append("")
    lines.append(f"- 全馬購入: n={baseline_all.n_bets:,}, 回収率={baseline_all.recovery_rate:.4f}")
    lines.append(f"- 1番人気全買い: n={baseline_fav.n_bets:,}, 回収率={baseline_fav.recovery_rate:.4f}, 的中率={baseline_fav.hit_rate:.4f}")
    lines.append("")
    lines.append("## グリッド結果（delta × min_n_eff）")
    lines.append("")
    lines.append("```text\n" + grid.to_string(index=False) + "\n```")
    lines.append("")
    lines.append("## 主要設定の分解評価")
    lines.append("")
    lines.append(
        f"n={primary_result.n_bets:,}, 投資={primary_result.investment:,.0f}円, "
        f"払戻={primary_result.payout:,.0f}円, 回収率={primary_result.recovery_rate:.4f}, "
        f"的中率={primary_result.hit_rate:.4f}, 最大DD={primary_result.max_drawdown:,.0f}円"
    )
    lines.append("")
    lines.append("### 年別")
    lines.append("")
    lines.append(fmt_table(primary_result.by_year))
    lines.append("")
    lines.append("### course_type別")
    lines.append("")
    lines.append(fmt_table(primary_result.by_course))
    lines.append("")
    lines.append("### 人気帯別")
    lines.append("")
    lines.append(fmt_table(primary_result.by_popularity_band))
    lines.append("")
    lines.append("## 検証コマンド")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m pytest tests/ -q")
    lines.append(f"python scripts/run_p0_baseline.py --results {args.results} --meta {args.meta}")
    lines.append("```")
    lines.append("")
    lines.append("## 成功/失敗")
    lines.append("")
    lines.append("成功条件（事前固定）: 主要設定で (a) 回収率 > 0.95 かつ1番人気ベースライン+5pt以上, (b) n_bets ≥ 500, (c) 年別回収率の符号一致 2/3以上, (d) 最大DD < 投資額30%。")
    lines.append("→ 上表から判定し、ここに結論を記入する。")
    lines.append("")
    lines.append("## Codexにレビューしてほしい点")
    lines.append("")
    lines.append("- trailing統計の同日除外（tests/test_trailing_stats.py の不変条件）")
    lines.append("- field_size_running 変更後の popularity_pct の妥当性")
    lines.append("- グリッド上の好成績セルだけを事後選択していないか（主要設定は事前指定）")

    report = "\n".join(lines)
    out_path = Path(args.out) if args.out else Path("results") / f"{today:%Y%m%d}_p0_win_calibration_backtest.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
