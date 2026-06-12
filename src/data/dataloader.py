"""Leak-safe data loading utilities for Pegasus-Edge.

The raw files contain one row per horse in ``race_results.csv`` and one row per
race in ``race_meta.csv``.  This module keeps post-race facts out of ``X`` and
returns labels/backtest returns separately in ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


RACE_ID = "race_id"
RACE_DATE = "kaisai_date"

REQUIRED_RESULT_COLUMNS: tuple[str, ...] = (
    "race_id",
    "着順",
    "枠番",
    "馬番",
    "馬名",
    "性齢",
    "斤量",
    "騎手",
    "タイム",
    "着差",
    "通過",
    "上り",
    "単勝",
    "人気",
    "馬体重",
    "調教師",
    "馬主",
    "賞金(万円)",
)

REQUIRED_META_COLUMNS: tuple[str, ...] = (
    "race_id",
    "kaisai_date",
    "course_type",
    "distance_m",
)

# A. Known before the race. Some of these are used only to derive normalized
# English feature columns or metadata, not necessarily passed directly to X.
PRE_RACE_COLUMNS: tuple[str, ...] = (
    "race_id",
    "kaisai_date",
    "course_type",
    "distance_m",
    "枠番",
    "馬番",
    "馬名",
    "性齢",
    "斤量",
    "騎手",
    "単勝",
    "人気",
    "馬体重",
    "調教師",
    "馬主",
)

# B. Facts generated during or after the race. Raw versions must never be in X.
POST_RACE_COLUMNS: tuple[str, ...] = (
    "タイム",
    "着差",
    "通過",
    "上り",
    "賞金(万円)",
)

# C. Labels/backtest outputs. Raw finish rank is also not allowed in X.
TARGET_COLUMNS: tuple[str, ...] = ("着順",)

PROHIBITED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    POST_RACE_COLUMNS
    + TARGET_COLUMNS
    + (
        "finish_position",
        "is_win",
        "is_top3",
        "win_payout_per_100yen",
        "win_profit_per_100yen",
        "race_time_seconds",
        "last3f",
        "prize_money",
        # src/features/past_run.py の中間post-race列 (shift(1)版のみ特徴量可)
        "time_behind_winner",
        "speed_resid",
        "last3f_rank_pct",
        "early_pos_pct",
        "late_pos_pct",
        "ground_gained",
        "race_strength",
    )
)

BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "frame_no",
    "horse_no",
    "assigned_weight",
    "win_odds",
    "popularity",
    "course_type",
    "distance_m",
    "race_month",
    "race_dayofweek",
    "sex",
    "age",
    "horse_weight_kg",
    "horse_weight_diff",
    "jockey",
    "trainer",
    "owner",
    "field_size",
    "field_size_running",
    "implied_win_prob",
    "market_prob",
    "popularity_pct",
    "log_win_odds",
)

HISTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "horse_entries_before",
    "horse_finished_starts_before",
    "horse_win_rate_before",
    "horse_top3_rate_before",
    "horse_avg_finish_before",
    "horse_last_finish_position",
    "horse_last_win_odds",
    "horse_last_popularity",
    "horse_last_distance_m",
    "horse_distance_change",
    "horse_days_since_last_start",
    "horse_last_time_seconds",
    "horse_avg_time_seconds_before",
    "horse_last_last3f",
    "horse_avg_last3f_before",
    "horse_last_course_type",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "race_id",
    "race_date",
    "horse_no",
    "horse_name",
    "field_size",
)

TARGET_OUTPUT_COLUMNS: tuple[str, ...] = (
    "finish_position",
    "is_win",
    "is_top3",
    "win_payout_per_100yen",
    "win_profit_per_100yen",
)


@dataclass(frozen=True)
class RaceDataset:
    """Aligned feature, target, and metadata frames."""

    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame


def load_dataset(
    results_path: str | Path = "race_results.csv",
    meta_path: str | Path = "race_meta.csv",
    *,
    merge_how: str = "left",
    include_history: bool = True,
    include_identifiers_in_X: bool = False,
    drop_invalid_outcomes: bool = True,
    drop_invalid_market: bool = True,
) -> RaceDataset:
    """Load raw CSVs and return leak-safe ``X``, ``y``, and row metadata.

    ``単勝`` is treated as a pre-race market signal and is also used to derive a
    unit-stake win payout because no explicit payout column exists in the raw
    files.  If a future scrape adds official refund columns, keep them in
    ``y``/backtest code only.
    """

    results, meta = load_raw_csvs(results_path, meta_path)
    merged = merge_results_with_meta(results, meta, how=merge_how)
    frame = prepare_model_frame(merged, include_history=include_history)

    if drop_invalid_outcomes:
        frame = frame.loc[frame["finish_position"].notna()].copy()

    if drop_invalid_market:
        frame = frame.loc[frame["win_odds"].notna() & frame["popularity"].notna()].copy()

    feature_columns = list(BASE_FEATURE_COLUMNS)
    if include_history:
        feature_columns.extend(HISTORY_FEATURE_COLUMNS)

    if include_identifiers_in_X:
        feature_columns = ["race_id", "race_date", "horse_name"] + feature_columns

    X = frame.loc[:, feature_columns].reset_index(drop=True)
    y = frame.loc[:, TARGET_OUTPUT_COLUMNS].reset_index(drop=True)
    metadata = frame.loc[:, METADATA_COLUMNS].reset_index(drop=True)

    assert_no_feature_leakage(X)
    return RaceDataset(X=X, y=y, metadata=metadata)


def load_prepared_frame(
    results_path: str | Path = "race_results.csv",
    meta_path: str | Path = "race_meta.csv",
    *,
    merge_how: str = "left",
    include_history: bool = True,
) -> pd.DataFrame:
    """Return the full normalized frame including outcome columns.

    Downstream feature builders and backtests need outcomes (``is_win``,
    ``win_profit_per_100yen``) next to pre-race columns to build trailing
    statistics.  Never feed this frame to a model directly; select feature
    columns and run :func:`assert_no_feature_leakage` on the selection.
    """

    results, meta = load_raw_csvs(results_path, meta_path)
    merged = merge_results_with_meta(results, meta, how=merge_how)
    return prepare_model_frame(merged, include_history=include_history)


def load_raw_csvs(
    results_path: str | Path = "race_results.csv",
    meta_path: str | Path = "race_meta.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and schema-check the two raw files."""

    results = pd.read_csv(results_path, low_memory=False)
    meta = pd.read_csv(meta_path, low_memory=False)

    _require_columns(results, REQUIRED_RESULT_COLUMNS, "race_results.csv")
    _require_columns(meta, REQUIRED_META_COLUMNS, "race_meta.csv")
    return results, meta


def merge_results_with_meta(
    results: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    how: str = "left",
    allow_missing_meta: bool = False,
) -> pd.DataFrame:
    """Merge horse rows with race metadata using a many-to-one race key."""

    if how not in {"left", "inner"}:
        raise ValueError("merge_how must be 'left' or 'inner'.")

    if meta[RACE_ID].duplicated().any():
        duplicated = meta.loc[meta[RACE_ID].duplicated(keep=False), RACE_ID].unique()[:10]
        raise ValueError(f"race_meta.csv has duplicated race_id values: {duplicated!r}")

    merged = results.merge(
        meta,
        on=RACE_ID,
        how=how,
        validate="many_to_one",
        indicator=True,
    )

    missing = merged["_merge"].eq("left_only")
    if missing.any() and not allow_missing_meta:
        sample = merged.loc[missing, RACE_ID].drop_duplicates().head(10).tolist()
        raise ValueError(f"{missing.sum()} result rows have no race_meta match. sample={sample}")

    return merged.drop(columns=["_merge"])


def prepare_model_frame(merged: pd.DataFrame, *, include_history: bool = True) -> pd.DataFrame:
    """Normalize raw columns and optionally add shifted historical features."""

    frame = merged.copy()
    frame["race_date"] = _parse_race_date(frame[RACE_DATE])
    frame["race_month"] = frame["race_date"].dt.month.astype("int16")
    frame["race_dayofweek"] = frame["race_date"].dt.dayofweek.astype("int16")

    frame["frame_no"] = pd.to_numeric(frame["枠番"], errors="coerce")
    frame["horse_no"] = pd.to_numeric(frame["馬番"], errors="coerce")
    frame["horse_name"] = frame["馬名"].astype("string")
    frame["assigned_weight"] = pd.to_numeric(frame["斤量"], errors="coerce")
    frame["jockey"] = frame["騎手"].astype("string")
    frame["trainer"] = frame["調教師"].astype("string")
    frame["owner"] = frame["馬主"].astype("string")
    frame["course_type"] = frame["course_type"].astype("string")
    frame["distance_m"] = pd.to_numeric(frame["distance_m"], errors="coerce")

    sex_age = frame["性齢"].astype("string").str.extract(r"^([^\d]+)(\d+)$")
    frame["sex"] = sex_age[0].astype("string")
    frame["age"] = pd.to_numeric(sex_age[1], errors="coerce")

    body = frame["馬体重"].astype("string")
    frame["horse_weight_kg"] = pd.to_numeric(body.str.extract(r"^(\d+)")[0], errors="coerce")
    frame["horse_weight_diff"] = pd.to_numeric(
        body.str.extract(r"\(([+-]?\d+)\)")[0],
        errors="coerce",
    )

    frame["finish_position"] = _parse_finish_position(frame["着順"])
    frame["is_win"] = frame["finish_position"].eq(1).fillna(False).astype("int8")
    frame["is_top3"] = frame["finish_position"].le(3).fillna(False).astype("int8")
    frame["race_time_seconds"] = _parse_time_to_seconds(frame["タイム"])
    frame["last3f"] = pd.to_numeric(frame["上り"], errors="coerce")
    frame["prize_money"] = pd.to_numeric(frame["賞金(万円)"], errors="coerce")

    frame["win_odds"] = pd.to_numeric(frame["単勝"], errors="coerce")
    frame["popularity"] = pd.to_numeric(frame["人気"], errors="coerce")
    frame["field_size"] = frame.groupby(RACE_ID)[RACE_ID].transform("size").astype("int16")
    # 取消・除外馬は単勝がNaNのまま行として残るため、実際に出走した頭数を別に持つ。
    # 人気は出走馬の中での順位なので、比率の分母は running 頭数に揃える。
    frame["field_size_running"] = (
        frame["win_odds"].notna().groupby(frame[RACE_ID]).transform("sum").astype("int16")
    )
    frame["implied_win_prob"] = 1.0 / frame["win_odds"]
    race_prob_sum = frame.groupby(RACE_ID)["implied_win_prob"].transform("sum")
    frame["market_prob"] = frame["implied_win_prob"] / race_prob_sum
    frame["popularity_pct"] = frame["popularity"] / frame["field_size_running"]
    frame["log_win_odds"] = np.log(frame["win_odds"])

    frame["win_payout_per_100yen"] = np.where(
        frame["is_win"].eq(1) & frame["win_odds"].notna(),
        frame["win_odds"] * 100.0,
        0.0,
    )
    frame["win_profit_per_100yen"] = frame["win_payout_per_100yen"] - 100.0

    if include_history:
        frame = add_horse_history_features(frame)

    return frame


def add_horse_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add horse-level historical facts shifted so current/future races are excluded."""

    ordered = frame.sort_values(["race_date", RACE_ID, "horse_no"]).copy()
    grouped = ordered.groupby("horse_name", sort=False)

    valid_finish = ordered["finish_position"].notna().astype(float)
    win = ordered["is_win"].astype(float)
    top3 = ordered["is_top3"].astype(float)
    finish_filled = ordered["finish_position"].fillna(0.0)
    time_valid = ordered["race_time_seconds"].notna().astype(float)
    time_filled = ordered["race_time_seconds"].fillna(0.0)
    last3f_valid = ordered["last3f"].notna().astype(float)
    last3f_filled = ordered["last3f"].fillna(0.0)

    ordered["_valid_finish"] = valid_finish
    ordered["_win"] = win
    ordered["_top3"] = top3
    ordered["_finish_filled"] = finish_filled
    ordered["_time_valid"] = time_valid
    ordered["_time_filled"] = time_filled
    ordered["_last3f_valid"] = last3f_valid
    ordered["_last3f_filled"] = last3f_filled

    ordered["horse_entries_before"] = grouped.cumcount()
    ordered["horse_finished_starts_before"] = (
        grouped["_valid_finish"].cumsum() - ordered["_valid_finish"]
    )
    wins_before = grouped["_win"].cumsum() - ordered["_win"]
    top3_before = grouped["_top3"].cumsum() - ordered["_top3"]
    finish_sum_before = grouped["_finish_filled"].cumsum() - ordered["_finish_filled"]
    time_count_before = grouped["_time_valid"].cumsum() - ordered["_time_valid"]
    time_sum_before = grouped["_time_filled"].cumsum() - ordered["_time_filled"]
    last3f_count_before = grouped["_last3f_valid"].cumsum() - ordered["_last3f_valid"]
    last3f_sum_before = grouped["_last3f_filled"].cumsum() - ordered["_last3f_filled"]

    starts = ordered["horse_finished_starts_before"].replace(0, np.nan)
    ordered["horse_win_rate_before"] = wins_before / starts
    ordered["horse_top3_rate_before"] = top3_before / starts
    ordered["horse_avg_finish_before"] = finish_sum_before / starts
    ordered["horse_avg_time_seconds_before"] = time_sum_before / time_count_before.replace(0, np.nan)
    ordered["horse_avg_last3f_before"] = last3f_sum_before / last3f_count_before.replace(0, np.nan)

    ordered["horse_last_finish_position"] = grouped["finish_position"].shift(1)
    ordered["horse_last_win_odds"] = grouped["win_odds"].shift(1)
    ordered["horse_last_popularity"] = grouped["popularity"].shift(1)
    ordered["horse_last_distance_m"] = grouped["distance_m"].shift(1)
    ordered["horse_distance_change"] = ordered["distance_m"] - ordered["horse_last_distance_m"]
    ordered["horse_last_time_seconds"] = grouped["race_time_seconds"].shift(1)
    ordered["horse_last_last3f"] = grouped["last3f"].shift(1)
    ordered["horse_last_course_type"] = grouped["course_type"].shift(1).astype("string")

    last_date = grouped["race_date"].shift(1)
    ordered["horse_days_since_last_start"] = (ordered["race_date"] - last_date).dt.days

    history = ordered.loc[:, list(HISTORY_FEATURE_COLUMNS)].reindex(frame.index)
    return pd.concat([frame.drop(columns=list(HISTORY_FEATURE_COLUMNS), errors="ignore"), history], axis=1)


def split_dataset_by_date(
    dataset: RaceDataset,
    test_start_date: str | pd.Timestamp,
    *,
    test_end_date: str | pd.Timestamp | None = None,
) -> tuple[RaceDataset, RaceDataset]:
    """Split an already loaded dataset by race date."""

    dates = pd.to_datetime(dataset.metadata["race_date"])
    start = pd.Timestamp(test_start_date)
    end = pd.Timestamp(test_end_date) if test_end_date is not None else None

    train_mask = dates < start
    test_mask = dates >= start
    if end is not None:
        test_mask &= dates <= end

    return _slice_dataset(dataset, train_mask), _slice_dataset(dataset, test_mask)


def iter_time_series_splits(
    race_dates: Sequence[object] | pd.Series,
    *,
    n_splits: int = 5,
    test_size_dates: int | None = None,
    min_train_dates: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window train/test indices over sorted unique race dates."""

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")

    dates = pd.Series(pd.to_datetime(race_dates), name="race_date")
    unique_dates = pd.Index(dates.dropna().sort_values().unique())
    if unique_dates.empty:
        raise ValueError("race_dates contains no valid dates.")

    if test_size_dates is None:
        test_size_dates = max(1, len(unique_dates) // (n_splits + 1))

    required_dates = min_train_dates + n_splits * test_size_dates
    if len(unique_dates) < required_dates:
        raise ValueError(
            "Not enough unique race dates for the requested split settings: "
            f"have={len(unique_dates)}, need={required_dates}."
        )

    first_test_start = len(unique_dates) - n_splits * test_size_dates
    for fold in range(n_splits):
        test_start_idx = first_test_start + fold * test_size_dates
        test_end_idx = test_start_idx + test_size_dates
        train_dates = unique_dates[:test_start_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        train_idx = dates.index[dates.isin(train_dates)].to_numpy()
        test_idx = dates.index[dates.isin(test_dates)].to_numpy()
        yield train_idx, test_idx


def assert_no_feature_leakage(X: pd.DataFrame) -> None:
    """Raise if raw post-race or target columns appear in the feature matrix."""

    leaked = sorted(PROHIBITED_FEATURE_COLUMNS.intersection(X.columns))
    if leaked:
        raise ValueError(f"Post-race/target columns leaked into X: {leaked}")


def _require_columns(df: pd.DataFrame, required: Sequence[str], file_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def _parse_race_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    dates = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if dates.isna().any():
        sample = series.loc[dates.isna()].head(10).tolist()
        raise ValueError(f"Invalid race dates found in kaisai_date. sample={sample}")
    return dates


def _parse_finish_position(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.extract(r"^(\d+)")[0]
    return pd.to_numeric(text, errors="coerce")


def _parse_time_to_seconds(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    split = text.str.extract(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
    minutes = pd.to_numeric(split[0], errors="coerce").fillna(0.0)
    seconds = pd.to_numeric(split[1], errors="coerce")
    return minutes * 60.0 + seconds


def _slice_dataset(dataset: RaceDataset, mask: pd.Series | np.ndarray) -> RaceDataset:
    mask_array = np.asarray(mask, dtype=bool)
    return RaceDataset(
        X=dataset.X.loc[mask_array].reset_index(drop=True),
        y=dataset.y.loc[mask_array].reset_index(drop=True),
        metadata=dataset.metadata.loc[mask_array].reset_index(drop=True),
    )


if __name__ == "__main__":
    ds = load_dataset()
    print(f"X shape: {ds.X.shape}")
    print(f"y shape: {ds.y.shape}")
    print(f"date range: {ds.metadata['race_date'].min()} -> {ds.metadata['race_date'].max()}")
