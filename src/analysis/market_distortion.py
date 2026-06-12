"""市場歪みの事前監査 (設計書 §8-5)。

ML以前に「大衆オッズの歪み（favorite-longshot bias 等）」が本データに
存在するかを直接測定する。人気帯・オッズ帯ごとに全馬単勝フラット買いの
素ROIを年別に集計し、モデルが拾うべきシグナルの所在地図を作る。

実行:
  py -m src.analysis.market_distortion
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataloader import load_dataset

ODDS_BANDS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf"))
POPULARITY_BANDS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 99))


def build_frame() -> pd.DataFrame:
    ds = load_dataset(include_history=False)
    df = pd.concat([ds.X, ds.y, ds.metadata[["race_id", "race_date"]]], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df["year"] = pd.to_datetime(df["race_date"]).dt.year
    return df


def roi_table(df: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    grouped = df.assign(_band=key).groupby(["year", "_band"], observed=True)
    out = grouped.agg(
        n=("is_win", "size"),
        win_rate=("is_win", "mean"),
        payout=("win_payout_per_100yen", "sum"),
    ).reset_index()
    out["roi"] = out["payout"] / (out["n"] * 100.0)
    out = out.rename(columns={"_band": label})
    return out


def pivot_roi(table: pd.DataFrame, label: str) -> pd.DataFrame:
    pivot = table.pivot(index=label, columns="year", values="roi")
    total = table.groupby(label, observed=True).agg(
        n=("n", "sum"), payout=("payout", "sum")
    )
    pivot["ALL"] = total["payout"] / (total["n"] * 100.0)
    pivot["n_total"] = total["n"]
    return pivot.round(4)


def main() -> None:
    df = build_frame()

    odds_band = pd.cut(df["win_odds"], bins=ODDS_BANDS, right=False)
    odds_table = roi_table(df, odds_band, "odds_band")

    pop_labels = pd.Series(pd.NA, index=df.index, dtype="string")
    for lo, hi in POPULARITY_BANDS:
        mask = df["popularity"].between(lo, hi)
        pop_labels[mask] = f"{lo}-{hi}人気" if lo != hi else f"{lo}人気"
    pop_table = roi_table(df, pop_labels, "pop_band")

    overall_roi = df["win_payout_per_100yen"].sum() / (len(df) * 100.0)
    implied_takeout_wall = 1.0 / df.groupby("race_id")["implied_win_prob"].sum().mean()

    print(f"全馬フラット買いROI: {overall_roi:.4f}")
    print(f"平均合成オッズ逆数 (1/B): {implied_takeout_wall:.4f}")
    print("\n=== オッズ帯別 ROI (年別) ===")
    odds_pivot = pivot_roi(odds_table, "odds_band")
    print(odds_pivot.to_string())
    print("\n=== 人気帯別 ROI (年別) ===")
    pop_pivot = pivot_roi(pop_table, "pop_band")
    print(pop_pivot.to_string())

    out_path = Path("results/20260612_market_distortion_audit.md")
    lines = [
        "# 市場歪み事前監査（人気帯・オッズ帯別 素ROI）",
        "",
        "- 日付: 2026-06-12",
        "- 目的: 設計書 §8-5。MLの前に favorite-longshot bias の存在と所在を実測する。",
        "- 方法: 全出走馬に単勝100円フラット買いした場合のROIを帯別×年別に集計。",
        f"- 全体ROI: **{overall_roi:.4f}**（理論値 1/B = {implied_takeout_wall:.4f} と整合するか要確認）",
        "",
        "## オッズ帯別 ROI",
        "",
        odds_pivot.to_markdown(),
        "",
        "## 人気帯別 ROI",
        "",
        pop_pivot.to_markdown(),
        "",
        "## 解釈メモ",
        "",
        "- 帯間でROIに系統差があれば、それが市場の歪み（モデルの獲物）の一次証拠。",
        "- 全帯がほぼ一様に 1/B 近傍なら、歪みは条件付き（特徴量依存）にのみ存在する。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポート保存: {out_path}")


if __name__ == "__main__":
    main()
