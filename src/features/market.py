"""Race-level market microstructure features (recipe group A1-A2).

All inputs are pre-race market quantities derived from 単勝 (treated as the
final pari-mutuel odds) and 人気.  Race-level aggregates are broadcast to
horse rows.  The only trailing statistic, ``overround_excess``, compares the
race overround against the expanding median of strictly earlier dates, so no
same-day or future information is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RACE_ID = "race_id"

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "race_overround",
    "overround_excess",
    "fav_odds",
    "fav_dominance",
    "gap_to_fav",
    "log_odds_gap_prev",
    "log_odds_gap_next",
    "market_entropy",
    "prob_x_entropy",
    "pop_pct_x_dominance",
)


def add_market_microstructure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add A1-A2 features. Requires the prepared-frame columns
    ``win_odds, log_win_odds, implied_win_prob, market_prob, popularity,
    popularity_pct, field_size_running, horse_no, race_date``."""

    out = frame.copy()
    valid = out["win_odds"].notna()

    # A1: race-level overround (sum of 1/odds over actual runners).
    out["race_overround"] = out.groupby(RACE_ID)["implied_win_prob"].transform("sum")

    # A2: favourite structure.
    out["fav_odds"] = out.groupby(RACE_ID)["win_odds"].transform("min")
    out["gap_to_fav"] = out["log_win_odds"] - np.log(out["fav_odds"])

    ranked = (
        out.loc[valid, [RACE_ID, "market_prob"]]
        .sort_values([RACE_ID, "market_prob"], ascending=[True, False])
        .assign(_rank=lambda d: d.groupby(RACE_ID).cumcount())
    )
    q1 = ranked.loc[ranked["_rank"].eq(0)].set_index(RACE_ID)["market_prob"]
    q2 = ranked.loc[ranked["_rank"].eq(1)].set_index(RACE_ID)["market_prob"]
    out["fav_dominance"] = out[RACE_ID].map(q1 / q2)

    # A2: local crowd density around each horse in popularity order.
    by_pop = out.loc[valid].sort_values([RACE_ID, "popularity", "win_odds", "horse_no"])
    out.loc[by_pop.index, "log_odds_gap_prev"] = by_pop.groupby(RACE_ID)["log_win_odds"].diff()
    out.loc[by_pop.index, "log_odds_gap_next"] = -by_pop.groupby(RACE_ID)["log_win_odds"].diff(-1)

    # A2: normalized market entropy (0 = single strong favourite, 1 = coin-flip field).
    qlogq = -(out["market_prob"] * np.log(out["market_prob"]))
    entropy = qlogq.groupby(out[RACE_ID]).transform("sum")
    denom = np.log(out["field_size_running"].where(out["field_size_running"] > 1))
    out["market_entropy"] = entropy / denom

    # A1: overround deviation vs the expanding median of earlier dates only.
    race_level = out.loc[valid, [RACE_ID, "race_date", "race_overround"]].drop_duplicates(RACE_ID)
    daily_mean = race_level.groupby("race_date")["race_overround"].mean().sort_index()
    trailing_median = daily_mean.expanding().median().shift(1)
    out["overround_excess"] = out["race_overround"] - out["race_date"].map(trailing_median)

    out["prob_x_entropy"] = out["market_prob"] * out["market_entropy"]
    out["pop_pct_x_dominance"] = out["popularity_pct"] * out["fav_dominance"]
    return out
