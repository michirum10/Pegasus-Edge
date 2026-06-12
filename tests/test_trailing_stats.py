import numpy as np
import pandas as pd
import pytest

from src.features.trailing_stats import add_odds_zone_calibration, add_trailing_group_stats

D0 = pd.Timestamp("2024-01-06")
D1 = D0 + pd.Timedelta(days=730)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_date": [D0, D0, D1],
            "zone": ["A", "A", "A"],
            "is_win": [1, 0, 1],
            "market_prob": [0.4, 0.2, 0.3],
            "win_profit_per_100yen": [300.0, -100.0, 100.0],
        }
    )


def test_same_day_rows_see_no_history():
    out = add_trailing_group_stats(toy_frame(), ["zone"], prefix="z")
    day0 = out.loc[out["race_date"].eq(D0)]
    assert day0["z_hist_n"].eq(0.0).all()
    assert day0["z_hist_win_rate"].isna().all()


def test_decay_is_half_after_one_half_life():
    out = add_trailing_group_stats(toy_frame(), ["zone"], prefix="z", half_life_days=730.0)
    later = out.loc[out["race_date"].eq(D1)].iloc[0]
    assert later["z_hist_n"] == pytest.approx(2 * 0.5)
    assert later["z_hist_wins"] == pytest.approx(1 * 0.5)
    assert later["z_hist_win_rate"] == pytest.approx(0.5)
    assert later["z_hist_q_mean"] == pytest.approx(0.3)
    assert later["z_hist_roi"] == pytest.approx((300.0 - 100.0) * 0.5 / (100.0 * 1.0))


def test_history_ignores_future_and_same_day_changes():
    base = add_trailing_group_stats(toy_frame(), ["zone"], prefix="z")
    mutated_frame = toy_frame()
    mutated_frame.loc[mutated_frame["race_date"].eq(D1), "is_win"] = 0
    mutated_frame.loc[mutated_frame["race_date"].eq(D1), "win_profit_per_100yen"] = -100.0
    mutated = add_trailing_group_stats(mutated_frame, ["zone"], prefix="z")
    hist_cols = [c for c in base.columns if c.startswith("z_hist_")]
    pd.testing.assert_frame_equal(base[hist_cols], mutated[hist_cols])


def test_groups_do_not_share_history():
    frame = toy_frame()
    frame.loc[2, "zone"] = "B"
    out = add_trailing_group_stats(frame, ["zone"], prefix="z")
    assert out.loc[2, "z_hist_n"] == 0.0


def test_calibration_falls_back_to_market_prob_without_history():
    frame = pd.DataFrame(
        {
            "race_date": [D0, D0],
            "win_odds": [2.0, 6.0],
            "is_win": [1, 0],
            "market_prob": [0.6, 0.2],
            "win_profit_per_100yen": [100.0, -100.0],
        }
    )
    out = add_odds_zone_calibration(frame)
    assert np.allclose(out["calib_p_win_zone"], out["market_prob"])
    assert out["calib_n_eff_zone"].eq(0.0).all()


def test_calibration_shrinkage_formula():
    frame = pd.DataFrame(
        {
            "race_date": [D0, D1],
            "win_odds": [2.5, 2.5],
            "is_win": [1, 0],
            "market_prob": [0.5, 0.5],
            "win_profit_per_100yen": [150.0, -100.0],
        }
    )
    out = add_odds_zone_calibration(frame, shrinkage_k=1.0, half_life_days=1e9)
    later = out.loc[out["race_date"].eq(D1)].iloc[0]
    # hist: n=1, wins=1, q_mean=0.5 -> (1 + 1*0.5) / (1 + 1) = 0.75
    assert later["calib_p_win_zone"] == pytest.approx(0.75, rel=1e-4)
    assert later["calib_hit_gap_zone"] == pytest.approx(0.5, rel=1e-4)
    # zone roi = 1.5, global roi = 1.5 -> shrunk stays 1.5
    assert later["calib_roi_zone"] == pytest.approx(1.5, rel=1e-4)
