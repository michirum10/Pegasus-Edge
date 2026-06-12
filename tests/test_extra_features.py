"""actor expanding 特徴量のリーク回帰テスト (Codexレビュー 2026-06-13 High指摘)。"""

import numpy as np
import pandas as pd

from src.data.extra_features import add_actor_history

D0 = pd.Timestamp("2024-01-06")
D1 = D0 + pd.Timedelta(days=7)


def toy_frame() -> pd.DataFrame:
    """調教師Tが同一レースに2頭 (馬番1が勝つ)、同日に別レース、翌週に1走。"""
    return pd.DataFrame(
        {
            "race_date": [D0, D0, D0, D1],
            "race_id": ["r1", "r1", "r2", "r3"],
            "horse_no": [1, 2, 1, 1],
            "jockey": ["J1", "J2", "J1", "J1"],
            "trainer": ["T", "T", "T", "T"],
            "is_win": [1, 0, 1, 0],
            "is_top3": [1, 0, 1, 1],
        }
    )


def test_same_race_rows_see_no_same_race_results():
    out = add_actor_history(toy_frame())
    same_race = out.loc[out["race_id"].eq("r1")]
    # 同一レース内の僚馬の結果 (馬番1の勝利) が馬番2に混入してはならない
    assert same_race["trainer_starts_before"].eq(0.0).all()
    assert same_race["trainer_win_rate_before"].isna().all()


def test_same_day_rows_see_no_same_day_results():
    out = add_actor_history(toy_frame())
    # 同日の別レース (r2) も当日結果を見てはならない
    r2 = out.loc[out["race_id"].eq("r2")].iloc[0]
    assert r2["trainer_starts_before"] == 0.0
    assert r2["jockey_starts_before"] == 0.0


def test_next_day_sees_full_previous_day():
    out = add_actor_history(toy_frame())
    r3 = out.loc[out["race_id"].eq("r3")].iloc[0]
    # 翌週は前日までの3走 (勝2) が見える
    assert r3["trainer_starts_before"] == 3.0
    assert r3["trainer_win_rate_before"] == 2.0 / 3.0
    assert r3["jockey_starts_before"] == 2.0
    assert r3["jockey_win_rate_before"] == 1.0


def test_future_results_do_not_change_history():
    base = add_actor_history(toy_frame())
    mutated_frame = toy_frame()
    mutated_frame.loc[3, "is_win"] = 1
    mutated = add_actor_history(mutated_frame)
    cols = [c for c in base.columns if c.endswith("_before")]
    pd.testing.assert_frame_equal(
        base.loc[base["race_date"].eq(D0), cols],
        mutated.loc[mutated["race_date"].eq(D0), cols],
    )


def test_row_order_and_length_preserved():
    frame = toy_frame().sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = add_actor_history(frame)
    assert len(out) == len(frame)
    assert (out["race_id"] == frame["race_id"]).all()
    assert (out["horse_no"] == frame["horse_no"]).all()
