import numpy as np
import pandas as pd
import pytest

from src.features.market import add_market_microstructure_features

D0 = pd.Timestamp("2024-01-06")
D1 = pd.Timestamp("2024-01-13")


def make_race(race_id: str, date: pd.Timestamp, odds: list[float]) -> pd.DataFrame:
    odds_arr = np.asarray(odds, dtype=float)
    implied = 1.0 / odds_arr
    market = implied / implied.sum()
    return pd.DataFrame(
        {
            "race_id": race_id,
            "race_date": date,
            "horse_no": np.arange(1, len(odds) + 1),
            "win_odds": odds_arr,
            "log_win_odds": np.log(odds_arr),
            "implied_win_prob": implied,
            "market_prob": market,
            "popularity": np.argsort(np.argsort(odds_arr)) + 1.0,
            "popularity_pct": (np.argsort(np.argsort(odds_arr)) + 1.0) / len(odds),
            "field_size_running": len(odds),
        }
    )


def test_uniform_race_microstructure():
    frame = make_race("R1", D0, [4.0, 4.0, 4.0, 4.0])
    out = add_market_microstructure_features(frame)
    assert np.allclose(out["market_entropy"], 1.0)
    assert np.allclose(out["gap_to_fav"], 0.0)
    assert np.allclose(out["fav_dominance"], 1.0)
    assert np.allclose(out["race_overround"], 1.0)
    assert out["log_odds_gap_prev"].isna().sum() == 1  # favourite has no horse above
    assert np.allclose(out["log_odds_gap_prev"].dropna(), 0.0)


def test_dominant_favorite_lowers_entropy_and_sets_gaps():
    frame = make_race("R2", D0, [1.5, 6.0, 6.0, 6.0])
    out = add_market_microstructure_features(frame)
    assert (out["market_entropy"] < 1.0).all()
    fav = out.loc[out["popularity"].eq(1.0)].iloc[0]
    assert fav["gap_to_fav"] == pytest.approx(0.0)
    assert fav["fav_dominance"] == pytest.approx((1 / 1.5) / (1 / 6.0))
    second = out.loc[out["popularity"].eq(2.0)].iloc[0]
    assert second["log_odds_gap_prev"] == pytest.approx(np.log(6.0) - np.log(1.5))


def test_overround_excess_uses_strictly_earlier_dates():
    day0 = make_race("R1", D0, [4.0, 4.0, 4.0, 4.0])  # overround 1.0
    day1 = make_race("R2", D1, [2.0, 3.0, 6.0, 6.0])  # overround 1.1667
    out = add_market_microstructure_features(pd.concat([day0, day1], ignore_index=True))
    assert out.loc[out["race_date"].eq(D0), "overround_excess"].isna().all()
    excess = out.loc[out["race_date"].eq(D1), "overround_excess"]
    assert np.allclose(excess, (1 / 2 + 1 / 3 + 1 / 6 + 1 / 6) - 1.0)
