import numpy as np
import pandas as pd
import pytest

from src.backtest.win_backtest import ev_bet_mask, max_drawdown, run_flat_win_backtest

D0 = pd.Timestamp("2024-01-06")
D1 = pd.Timestamp("2025-01-13")


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2"],
            "race_date": [D0, D0, D1],
            "horse_no": [1, 2, 1],
            "win_odds": [4.0, 10.0, 2.0],
            "is_win": [1, 0, 0],
            "course_type": ["turf", "turf", "dirt"],
            "popularity": [2.0, 5.0, 1.0],
        }
    )


def test_flat_backtest_hand_math():
    frame = make_frame()
    mask = pd.Series([True, False, True], index=frame.index)
    result = run_flat_win_backtest(frame, mask)
    assert result.n_bets == 2
    assert result.investment == pytest.approx(200.0)
    assert result.payout == pytest.approx(400.0)
    assert result.recovery_rate == pytest.approx(2.0)
    assert result.roi == pytest.approx(1.0)
    assert result.hit_rate == pytest.approx(0.5)
    # profits ordered by date: [+300, -100] -> cum [300, 200] -> dd = 100
    assert result.max_drawdown == pytest.approx(100.0)
    assert result.by_year.loc[2024, "recovery_rate"] == pytest.approx(4.0)
    assert result.by_course.loc["dirt", "recovery_rate"] == pytest.approx(0.0)
    assert result.by_popularity_band.loc["1", "n_bets"] == 1


def test_max_drawdown_with_initial_losses():
    assert max_drawdown(np.array([-100.0, -100.0, 300.0])) == pytest.approx(200.0)
    assert max_drawdown(np.array([])) == 0.0


def test_ev_bet_mask_threshold_and_nan():
    frame = pd.DataFrame(
        {
            "win_odds": [2.6, 2.4, 2.6, 2.6],
            "calib_p_win_zone": [0.5, 0.5, 0.5, np.nan],
            "calib_n_eff_zone": [2000.0, 2000.0, 10.0, 2000.0],
        }
    )
    mask = ev_bet_mask(frame, delta=0.2, min_n_eff=1000.0)
    assert mask.tolist() == [True, False, False, False]
