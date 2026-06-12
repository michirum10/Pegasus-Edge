"""現存データのみで作る追加特徴量 (IG>0 達成のための特徴量強化 v2)。

- 騎手・調教師の expanding 成績: **厳密に前日までの日次集計**のみを使う。
  行単位の cumsum - current 方式は、同一レースに同じ調教師の馬が複数頭
  いる場合に同一レース内の結果が後続行へ混入し、同日他レースの結果も
  混入するため使用しない (Codexレビュー 2026-06-13 High指摘で修正)。
  日次セマンティクスは src/features/trailing_stats.py と同一。
- processed_races.csv からの racecourse / track_condition 結合
  (track_condition は数値コードのまま category として扱う。
   コード対応表は未検証だが、カテゴリ特徴量として使う分には対応不要)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACTOR_FEATURE_COLUMNS = (
    "jockey_starts_before",
    "jockey_win_rate_before",
    "jockey_top3_rate_before",
    "trainer_starts_before",
    "trainer_win_rate_before",
    "trainer_top3_rate_before",
)

RACE_CONTEXT_COLUMNS = ("racecourse", "track_condition")


def _daily_actor_stats(df: pd.DataFrame, actor_col: str, prefix: str) -> pd.DataFrame:
    """actor×日付で集計し、厳密に前日までの累積成績を返す。"""
    keys = [actor_col, "race_date"]
    daily = (
        df.groupby(keys, observed=True, dropna=True)
        .agg(_n=("is_win", "size"), _wins=("is_win", "sum"), _top3=("is_top3", "sum"))
        .reset_index()
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    grouped = daily.groupby(actor_col, observed=True, sort=False)
    n_before = (grouped["_n"].cumsum() - daily["_n"]).astype("float64")
    wins_before = grouped["_wins"].cumsum() - daily["_wins"]
    top3_before = grouped["_top3"].cumsum() - daily["_top3"]
    denom = n_before.replace(0.0, np.nan)
    daily[f"{prefix}_starts_before"] = n_before
    daily[f"{prefix}_win_rate_before"] = wins_before / denom
    daily[f"{prefix}_top3_rate_before"] = top3_before / denom
    feature_cols = [c for c in daily.columns if c.startswith(f"{prefix}_")]
    return daily[keys + feature_cols]


def add_actor_history(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の expanding 成績列 (前日まで) を追加して返す。"""
    out = df
    for actor_col, prefix in (("jockey", "jockey"), ("trainer", "trainer")):
        stats = _daily_actor_stats(out, actor_col, prefix)
        out = out.merge(stats, on=[actor_col, "race_date"], how="left",
                        validate="many_to_one")
    return out


def add_race_context(
    df: pd.DataFrame, processed_path: str | Path = "processed_races.csv"
) -> pd.DataFrame:
    """processed_races.csv の競馬場・馬場状態をレース単位で結合する。"""
    context = pd.read_csv(
        processed_path, usecols=["race_id", "racecourse", "track_condition"]
    ).drop_duplicates("race_id")
    merged = df.merge(context, on="race_id", how="left", validate="many_to_one")
    n_missing = merged["racecourse"].isna().sum()
    if n_missing:
        print(f"race context 欠落行: {n_missing} ({n_missing / len(merged):.2%})")
    return merged
