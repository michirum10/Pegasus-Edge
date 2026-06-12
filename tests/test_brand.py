import numpy as np
import pandas as pd
import pytest

from src.features.brand import BRAND_FEATURE_COLUMNS, add_brand_overbet_features

D0 = pd.Timestamp("2024-01-06")
D1 = pd.Timestamp("2024-01-13")


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_date": [D0, D0, D0, D1, D1],
            "jockey": ["A", "A", "B", "A", "B"],
            "trainer": ["T1", "T1", "T2", "T1", "T2"],
            "is_win": [1, 0, 0, 0, 0],
            "market_prob": [0.4, 0.2, 0.3, 0.3, 0.3],
            "win_profit_per_100yen": [300.0, -100.0, -100.0, -100.0, -100.0],
        }
    )


def test_first_date_defaults():
    out = add_brand_overbet_features(toy_frame(), half_life_days=1e9, shrinkage_k=2.0)
    day0 = out.loc[out["race_date"].eq(D0)]
    assert day0["jockey_overbet"].eq(0.0).all()
    assert day0["jockey_n_eff"].eq(0.0).all()
    assert day0["jockey_roi"].isna().all()  # 全体ROIも履歴なし


def test_overbet_and_roi_hand_math():
    out = add_brand_overbet_features(toy_frame(), half_life_days=1e9, shrinkage_k=2.0)
    a1 = out.loc[out["race_date"].eq(D1) & out["jockey"].eq("A")].iloc[0]
    # jockey A 履歴: n=2, q_mean=0.3, win_rate=0.5 -> gap=-0.2 (過小評価されている)
    assert a1["jockey_overbet"] == pytest.approx(2 * (-0.2) / (2 + 2), rel=1e-4)
    # A の ROI=1.0, 全体 ROI=(300-100-100)/300=1/3 -> (2*1.0 + 2/3)/4
    assert a1["jockey_roi"] == pytest.approx((2 * 1.0 + 2 * (1 / 3)) / 4, rel=1e-4)
    assert a1["jockey_n_eff"] == pytest.approx(2.0, rel=1e-4)
    b1 = out.loc[out["race_date"].eq(D1) & out["jockey"].eq("B")].iloc[0]
    # jockey B 履歴: n=1, q_mean=0.3, win_rate=0 -> gap=+0.3 (過大評価=overbet)
    assert b1["jockey_overbet"] == pytest.approx(1 * 0.3 / (1 + 2), rel=1e-4)


def test_helper_columns_dropped_and_features_present():
    out = add_brand_overbet_features(toy_frame())
    for col in BRAND_FEATURE_COLUMNS:
        assert col in out.columns
    assert not [c for c in out.columns if c.startswith(("_bglob_", "_jockey_", "_trainer_"))]


def test_future_changes_do_not_affect_history():
    base = add_brand_overbet_features(toy_frame(), half_life_days=1e9, shrinkage_k=2.0)
    mutated_frame = toy_frame()
    mutated_frame.loc[mutated_frame["race_date"].eq(D1), "is_win"] = 1
    mutated_frame.loc[mutated_frame["race_date"].eq(D1), "win_profit_per_100yen"] = 900.0
    mutated = add_brand_overbet_features(mutated_frame, half_life_days=1e9, shrinkage_k=2.0)
    cols = list(BRAND_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(base[cols], mutated[cols])
