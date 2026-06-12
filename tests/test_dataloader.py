import numpy as np
import pytest

from tests.synth import meta_row, prepared_frame, result_row


@pytest.fixture()
def frame_with_scratch():
    rows = [
        result_row("R1", 1, odds=2.0, popularity=1, finish=1),
        result_row("R1", 2, odds=4.0, popularity=2, finish=2),
        result_row("R1", 3, odds=8.0, popularity=3, finish=3),
        result_row("R1", 4, odds=None, popularity=None, finish="取"),
    ]
    return prepared_frame(rows, [meta_row("R1", "20240106")])


def test_field_size_counts_all_rows(frame_with_scratch):
    assert frame_with_scratch["field_size"].eq(4).all()


def test_field_size_running_excludes_scratched(frame_with_scratch):
    assert frame_with_scratch["field_size_running"].eq(3).all()


def test_market_prob_sums_to_one_over_runners(frame_with_scratch):
    total = frame_with_scratch["market_prob"].sum()
    assert total == pytest.approx(1.0)
    assert frame_with_scratch["market_prob"].isna().sum() == 1


def test_popularity_pct_uses_running_field_size(frame_with_scratch):
    fav = frame_with_scratch.loc[frame_with_scratch["popularity"].eq(1)].iloc[0]
    assert fav["popularity_pct"] == pytest.approx(1.0 / 3.0)


def test_win_payout_approximation(frame_with_scratch):
    winner = frame_with_scratch.loc[frame_with_scratch["finish_position"].eq(1)].iloc[0]
    assert winner["win_payout_per_100yen"] == pytest.approx(200.0)
    assert winner["win_profit_per_100yen"] == pytest.approx(100.0)
    losers = frame_with_scratch.loc[frame_with_scratch["finish_position"].gt(1)]
    assert np.allclose(losers["win_profit_per_100yen"], -100.0)
