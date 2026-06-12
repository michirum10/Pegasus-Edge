"""Minimal flat-stake win-bet backtest (CLAUDE.md §5.3 metrics).

Payouts follow the loader's pari-mutuel approximation: a winning 100yen bet
returns ``win_odds * 100`` because no official refund table exists yet in the
raw data.  ``単勝`` is the final odds, so both the bet decision and the payout
are evaluated at the same self-consistent price; replace with the payout cache
(recipe C5) once its schema audit lands.

回収率 (recovery_rate) = payout / investment, ROI = recovery_rate - 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POPULARITY_BAND_EDGES: tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 10.0, float("inf"))


@dataclass(frozen=True)
class WinBacktestResult:
    n_bets: int
    investment: float
    payout: float
    profit: float
    recovery_rate: float
    roi: float
    hit_rate: float
    max_drawdown: float
    by_year: pd.DataFrame = field(repr=False)
    by_course: pd.DataFrame = field(repr=False)
    by_popularity_band: pd.DataFrame = field(repr=False)


def ev_bet_mask(
    frame: pd.DataFrame,
    *,
    p_col: str = "calib_p_win_zone",
    n_eff_col: str = "calib_n_eff_zone",
    delta: float = 0.2,
    min_n_eff: float = 1000.0,
) -> pd.Series:
    """Bet when estimated EV = p_hat * odds exceeds 1 + delta with enough history."""

    ev = frame[p_col] * frame["win_odds"]
    mask = (ev > 1.0 + delta) & (frame[n_eff_col] >= min_n_eff)
    return mask.fillna(False)


def run_flat_win_backtest(
    frame: pd.DataFrame,
    bet_mask: pd.Series,
    *,
    stake: float = 100.0,
) -> WinBacktestResult:
    """Evaluate flat-stake win bets on the rows selected by ``bet_mask``.

    ``frame`` needs ``race_id, race_date, horse_no, win_odds, is_win,
    course_type, popularity``.
    """

    bets = frame.loc[bet_mask.fillna(False) & frame["win_odds"].notna()].copy()
    bets = bets.sort_values(["race_date", "race_id", "horse_no"], kind="stable")

    bets["_payout"] = np.where(bets["is_win"].eq(1), bets["win_odds"] * stake, 0.0)
    bets["_profit"] = bets["_payout"] - stake

    n_bets = len(bets)
    investment = float(n_bets * stake)
    payout = float(bets["_payout"].sum())
    profit = payout - investment
    recovery = payout / investment if investment > 0 else float("nan")
    hit_rate = float(bets["is_win"].mean()) if n_bets else float("nan")

    return WinBacktestResult(
        n_bets=n_bets,
        investment=investment,
        payout=payout,
        profit=profit,
        recovery_rate=recovery,
        roi=recovery - 1.0 if investment > 0 else float("nan"),
        hit_rate=hit_rate,
        max_drawdown=max_drawdown(bets["_profit"].to_numpy()),
        by_year=_decompose(bets, bets["race_date"].dt.year.rename("year"), stake),
        by_course=_decompose(bets, bets["course_type"].rename("course_type"), stake),
        by_popularity_band=_decompose(bets, _popularity_band(bets), stake),
    )


def max_drawdown(profit_sequence: np.ndarray) -> float:
    """Peak-to-trough drop of the cumulative profit, in stake currency."""

    if len(profit_sequence) == 0:
        return 0.0
    cumulative = np.cumsum(profit_sequence)
    running_peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    return float(np.max(running_peak - cumulative))


def _popularity_band(bets: pd.DataFrame) -> pd.Series:
    band = pd.cut(
        bets["popularity"],
        list(POPULARITY_BAND_EDGES),
        right=False,
        labels=["1", "2-3", "4-6", "7-9", "10+"],
    )
    return band.astype("string").rename("popularity_band")


def _decompose(bets: pd.DataFrame, key: pd.Series, stake: float) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(
            columns=["n_bets", "investment", "payout", "recovery_rate", "hit_rate"]
        )
    grouped = bets.groupby(key, observed=True)
    table = grouped.agg(n_bets=("_payout", "size"), payout=("_payout", "sum"), hits=("is_win", "sum"))
    table["investment"] = table["n_bets"] * stake
    table["recovery_rate"] = table["payout"] / table["investment"]
    table["hit_rate"] = table["hits"] / table["n_bets"]
    return table[["n_bets", "investment", "payout", "recovery_rate", "hit_rate"]]
