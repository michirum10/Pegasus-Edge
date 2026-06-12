"""Leak-safe trailing outcome statistics (report 20260612 §3.0, recipe A3).

Historical encodings (zone calibration, brand overbet indices, EV maps) must
never see the current race day: races on the same ``kaisai_date`` can finish
after the bet is placed and per-race start times are not available.  This
module therefore aggregates outcomes per calendar date and exposes, for each
row, exponentially time-decayed sums over strictly earlier dates only.

Empirical-Bayes shrinkage toward a trailing global prior keeps thin cells from
producing fake edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE_COL = "race_date"

# 単勝1.0倍はあり得るため左端は1.0、右端は事実上の上限なし。
DEFAULT_ODDS_ZONE_EDGES: tuple[float, ...] = (
    1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 100.0, float("inf"),
)

CALIBRATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "calib_p_win_zone",
    "calib_hit_gap_zone",
    "calib_roi_zone",
    "calib_n_eff_zone",
)


def add_trailing_group_stats(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    prefix: str,
    half_life_days: float = 730.0,
    win_col: str = "is_win",
    market_prob_col: str = "market_prob",
    profit_col: str = "win_profit_per_100yen",
) -> pd.DataFrame:
    """Attach decayed sums over strictly earlier dates per group.

    Adds ``{prefix}_hist_n`` (decayed observation count), ``{prefix}_hist_wins``,
    ``{prefix}_hist_win_rate``, ``{prefix}_hist_q_mean`` (mean market prob) and
    ``{prefix}_hist_roi`` (mean profit per 100yen stake / 100).

    The caller must pass rows already filtered to valid outcomes
    (``finish_position`` and ``win_odds`` not null); every row counts as one
    flat 100yen bet in the historical ROI.
    """

    keys = list(group_cols) + [DATE_COL]
    daily = (
        frame.groupby(keys, observed=True, dropna=True)
        .agg(
            _n=(win_col, "size"),
            _wins=(win_col, "sum"),
            _q=(market_prob_col, "sum"),
            _profit=(profit_col, "sum"),
        )
        .reset_index()
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )

    if group_cols:
        same_as_prev = np.array(
            (daily[group_cols] == daily[group_cols].shift()).all(axis=1), dtype=bool
        )
    else:
        same_as_prev = np.ones(len(daily), dtype=bool)
    if len(same_as_prev):
        same_as_prev[0] = False

    dates = daily[DATE_COL].to_numpy(dtype="datetime64[ns]")
    today = daily[["_n", "_wins", "_q", "_profit"]].to_numpy(dtype=float)
    hist = np.zeros_like(today)
    state = np.zeros(4)
    prev_date = None
    for i in range(len(daily)):
        if not same_as_prev[i]:
            state = np.zeros(4)
            prev_date = None
        if prev_date is not None:
            delta_days = (dates[i] - prev_date) / np.timedelta64(1, "D")
            state = state * (0.5 ** (delta_days / half_life_days))
        hist[i] = state
        state = state + today[i]
        prev_date = dates[i]

    n_hist = hist[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        daily[f"{prefix}_hist_n"] = n_hist
        daily[f"{prefix}_hist_wins"] = hist[:, 1]
        daily[f"{prefix}_hist_win_rate"] = np.where(n_hist > 0, hist[:, 1] / n_hist, np.nan)
        daily[f"{prefix}_hist_q_mean"] = np.where(n_hist > 0, hist[:, 2] / n_hist, np.nan)
        daily[f"{prefix}_hist_roi"] = np.where(n_hist > 0, hist[:, 3] / (100.0 * n_hist), np.nan)

    out_cols = keys + [c for c in daily.columns if c.startswith(f"{prefix}_hist_")]
    return frame.merge(daily[out_cols], on=keys, how="left")


def add_odds_zone_calibration(
    frame: pd.DataFrame,
    *,
    half_life_days: float = 730.0,
    shrinkage_k: float = 200.0,
    zone_edges: tuple[float, ...] = DEFAULT_ODDS_ZONE_EDGES,
) -> pd.DataFrame:
    """Recipe A3: walk-forward favorite-longshot calibration features.

    - ``calib_p_win_zone``: historical win rate of the horse's odds zone,
      shrunk toward the zone's own historical mean market probability, so an
      unseen zone implies "trust the market" (EV < 1, abstain).
    - ``calib_hit_gap_zone``: historical (win rate - market prob) of the zone.
    - ``calib_roi_zone``: historical flat-bet ROI of the zone, shrunk toward
      the trailing global ROI.
    - ``calib_n_eff_zone``: decayed effective sample size of the zone.
    """

    out = frame.copy()
    zone = pd.cut(out["win_odds"], list(zone_edges), right=False)
    out["odds_zone"] = zone.astype("string")

    out = add_trailing_group_stats(
        out, ["odds_zone"], prefix="zone", half_life_days=half_life_days
    )
    out = add_trailing_group_stats(out, [], prefix="glob", half_life_days=half_life_days)

    n = out["zone_hist_n"]
    k = float(shrinkage_k)
    p_shrunk = (out["zone_hist_wins"] + k * out["zone_hist_q_mean"]) / (n + k)
    out["calib_p_win_zone"] = p_shrunk.where(n > 0, out["market_prob"])
    out["calib_hit_gap_zone"] = out["zone_hist_win_rate"] - out["zone_hist_q_mean"]
    roi_shrunk = (n * out["zone_hist_roi"] + k * out["glob_hist_roi"]) / (n + k)
    out["calib_roi_zone"] = roi_shrunk
    out["calib_n_eff_zone"] = n.fillna(0.0)
    return out
