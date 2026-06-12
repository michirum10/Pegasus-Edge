import numpy as np
import pandas as pd
import pytest

from src.data.dataloader import assert_no_feature_leakage
from src.features.past_run import PAST_RUN_FEATURE_COLUMNS, add_past_run_features
from tests.synth import meta_row, prepared_frame, result_row


def two_race_history() -> tuple[list[dict], list[dict]]:
    r1 = [
        result_row("R1", 1, odds=2.0, popularity=1, finish=1, name="H1", time="1:36.0", agari="35.5", 通過="1-1"),
        result_row("R1", 2, odds=4.0, popularity=2, finish=2, name="H2", time="1:36.5", agari="35.0", 通過="2-2"),
        result_row("R1", 3, odds=6.0, popularity=3, finish=3, name="H3", time="1:37.0", agari="36.0", 通過="3-3"),
        result_row("R1", 4, odds=8.0, popularity=4, finish=4, name="H4", time="1:37.5", agari="36.5", 通過="4-4"),
        result_row("R1", 5, odds=10.0, popularity=5, finish=6, name="H5", time="1:38.2", agari="36.6", 通過="5-5"),
        result_row("R1", 6, odds=20.0, popularity=6, finish=7, name="H6", time="1:38.5", agari="36.7", 通過="6-6"),
        result_row("R1", 7, odds=30.0, popularity=7, finish=5, name="H7", time="1:38.0", agari="34.5", 通過="7-4"),
        result_row("R1", 8, odds=50.0, popularity=8, finish=8, name="H8", time="1:39.0", agari="37.0", 通過="8-8"),
    ]
    r2 = [
        result_row("R2", 1, odds=1.8, popularity=1, finish=1, name="H1"),
        result_row("R2", 2, odds=3.5, popularity=2, finish=2, name="H2"),
        result_row("R2", 3, odds=6.0, popularity=3, finish=3, name="H7"),
        result_row("R2", 4, odds=12.0, popularity=4, finish=4, name="H8"),
    ]
    metas = [meta_row("R1", "20240106"), meta_row("R2", "20240113")]
    return r1 + r2, metas


@pytest.fixture()
def features() -> pd.DataFrame:
    rows, metas = two_race_history()
    return add_past_run_features(prepared_frame(rows, metas))


def row(frame: pd.DataFrame, race: str, name: str) -> pd.Series:
    sel = frame.loc[frame["race_id"].eq(race) & frame["horse_name"].eq(name)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_first_start_has_no_history(features):
    first = features.loc[features["race_id"].eq("R1")]
    assert first["last_time_behind_winner"].isna().all()
    assert first["closer_excuse"].eq(0).all()
    assert first["excuse_score"].eq(0).all()


def test_time_behind_winner_shifted(features):
    assert row(features, "R2", "H1")["last_time_behind_winner"] == pytest.approx(0.0)
    assert row(features, "R2", "H2")["last_time_behind_winner"] == pytest.approx(0.5)
    assert row(features, "R2", "H7")["last_time_behind_winner"] == pytest.approx(2.0)


def test_last3f_rank_within_last_race(features):
    assert row(features, "R2", "H7")["last_last3f_rank_pct"] == pytest.approx(1 / 8)
    assert row(features, "R2", "H2")["last_last3f_rank_pct"] == pytest.approx(2 / 8)
    assert row(features, "R2", "H7")["last_last3f_top2"] == 1.0
    assert row(features, "R2", "H1")["last_last3f_top2"] == 0.0


def test_corner_parse_and_ground_gained(features):
    h7 = row(features, "R2", "H7")
    assert h7["last_early_pos_pct"] == pytest.approx(7 / 8)
    assert h7["last_late_pos_pct"] == pytest.approx(4 / 8)
    assert h7["last_ground_gained"] == pytest.approx(3 / 8)


def test_excuse_flags(features):
    assert row(features, "R2", "H7")["closer_excuse"] == 1   # 上り最速・後方・5着
    assert row(features, "R2", "H2")["closer_excuse"] == 0   # 上り2位だが先行
    assert row(features, "R2", "H8")["excusable_loss_draw"] == 1  # 大外・8着
    assert row(features, "R2", "H7")["excusable_loss_draw"] == 0  # 大外だが5着
    assert row(features, "R2", "H1")["flattered_win_inside"] == 1  # 最内・1着


def test_last_race_strength_is_top3_market_prob_mean(features):
    r1 = features.loc[features["race_id"].eq("R1")]
    expected = r1.loc[r1["finish_position"].le(3), "market_prob"].mean()
    r2 = features.loc[features["race_id"].eq("R2")]
    assert np.allclose(r2["last_race_strength"], expected)


def test_overreaction_index_sign(features):
    h7 = row(features, "R2", "H7")
    # R1: 7番人気/8頭=0.875 -> R2: 3番人気/4頭=0.75。人気を上げたので負
    assert h7["pop_drift_pct"] == pytest.approx(0.75 - 0.875)
    assert h7["excuse_score"] == 1
    assert h7["overreaction_index"] == pytest.approx(-0.125)


def test_current_race_outcome_does_not_change_features():
    rows, metas = two_race_history()
    base = add_past_run_features(prepared_frame(rows, metas))
    mutated_rows = [dict(r) for r in rows]
    for r in mutated_rows:
        if r["race_id"] == "R2":  # R2の結果を全て改変
            r["着順"] = 5 - int(r["着順"]) if isinstance(r["着順"], int) else r["着順"]
            r["タイム"] = "1:50.0"
            r["上り"] = "39.9"
            r["通過"] = "1-1"
    mutated = add_past_run_features(prepared_frame(mutated_rows, metas))
    cols = list(PAST_RUN_FEATURE_COLUMNS)
    r2_mask = base["race_id"].eq("R2").to_numpy()
    pd.testing.assert_frame_equal(
        base.loc[r2_mask, cols].reset_index(drop=True),
        mutated.loc[r2_mask, cols].reset_index(drop=True),
    )


def test_speed_resid_uses_trailing_reference():
    rows = [
        result_row("RA", 1, odds=2.0, popularity=1, finish=1, name="X", time="1:36.0"),
        result_row("RA", 2, odds=3.0, popularity=2, finish=2, name="Y", time="1:38.0"),
        result_row("RB", 1, odds=2.0, popularity=1, finish=1, name="X", time="1:40.0"),
        result_row("RB", 2, odds=3.0, popularity=2, finish=2, name="Z", time="1:42.0"),
        result_row("RC", 1, odds=2.0, popularity=1, finish=1, name="X", time="1:37.0"),
        result_row("RC", 2, odds=3.0, popularity=2, finish=2, name="W", time="1:39.0"),
    ]
    metas = [meta_row("RA", "20240106"), meta_row("RB", "20240113"), meta_row("RC", "20240120")]
    out = add_past_run_features(prepared_frame(rows, metas))
    # RA は基準なし -> RB 時点の last_speed_resid は NaN
    rb_x = out.loc[out["race_id"].eq("RB") & out["horse_name"].eq("X")].iloc[0]
    assert pd.isna(rb_x["last_speed_resid"])
    # RB の基準 = RA の日次中央値 97.0 -> X の RB 残差 = 100-97 = +3.0 が RC で見える
    rc_x = out.loc[out["race_id"].eq("RC") & out["horse_name"].eq("X")].iloc[0]
    assert rc_x["last_speed_resid"] == pytest.approx(3.0)


def test_no_post_race_helpers_leak_into_output():
    rows, metas = two_race_history()
    out = add_past_run_features(prepared_frame(rows, metas))
    assert_no_feature_leakage(out[list(PAST_RUN_FEATURE_COLUMNS)])
    for helper in ("_last3f_top2", "_horse_no_pct", "_dist_band"):
        assert helper not in out.columns
