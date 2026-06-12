"""現存データのみで作る追加特徴量 (IG>0 達成のための特徴量強化 v2)。

- 騎手・調教師の expanding 成績: 当該レースより前の行のみを集計
  (cumsum から自レース分を引く方式。dataloader の馬履歴と同じ規約)
- processed_races.csv からの racecourse / track_condition 結合
  (track_condition は数値コードのまま category として扱う。
   コード対応表は未検証だが、カテゴリ特徴量として使う分には対応不要)

リーク注意: 呼び出し側のフレームは (race_date, race_id, horse_no) で
ソート済みであること。expanding 集計はその行順を前提とする。
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


def _expanding_actor_stats(df: pd.DataFrame, actor_col: str, prefix: str) -> pd.DataFrame:
    if not df["race_date"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by race_date before expanding stats")
    grouped = df.groupby(actor_col, sort=False, observed=True)
    starts = grouped.cumcount().astype("float64")
    wins_before = grouped["is_win"].cumsum() - df["is_win"]
    top3_before = grouped["is_top3"].cumsum() - df["is_top3"]
    denom = starts.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            f"{prefix}_starts_before": starts,
            f"{prefix}_win_rate_before": wins_before / denom,
            f"{prefix}_top3_rate_before": top3_before / denom,
        },
        index=df.index,
    )


def add_actor_history(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の expanding 成績列を追加して返す。"""
    jockey = _expanding_actor_stats(df, "jockey", "jockey")
    trainer = _expanding_actor_stats(df, "trainer", "trainer")
    return pd.concat([df, jockey, trainer], axis=1)


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
