"""Past-run rollup features (recipe group B): de-noising the last finish.

The crowd over-reacts to the raw last finish position.  These features rebuild
what the finish position hides — beaten margin in seconds, in-race closing
rank, trip and draw excuses, field strength — strictly from each horse's
*previous* starts via ``shift(1)`` per horse, mirroring
``add_horse_history_features`` in the dataloader.

Contract: pass rows already filtered to valid outcomes (``finish_position``
and ``win_odds`` not null) so scratched entries do not count as past starts.
Intermediate post-race helper columns (``time_behind_winner`` etc.) are listed
in ``PROHIBITED_FEATURE_COLUMNS`` and never returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RACE_ID = "race_id"
SORT_KEYS = ["race_date", RACE_ID, "horse_no"]

PAST_RUN_FEATURE_COLUMNS: tuple[str, ...] = (
    # B1: time behind the winner (continuous beaten margin)
    "last_time_behind_winner",
    "best_time_behind_winner_career",
    "avg_time_behind_winner_3",
    # B2: speed-figure-lite residual vs trailing course/distance reference
    "last_speed_resid",
    "best_speed_resid_career",
    "speed_resid_trend",
    # B3: closing speed rank within the last race
    "last_last3f_rank_pct",
    "last_last3f_top2",
    # B4: position in running (corner parse)
    "last_early_pos_pct",
    "last_late_pos_pct",
    "last_ground_gained",
    # B5: draw context of the last race
    "last_horse_no_pct",
    "closer_excuse",
    "excusable_loss_draw",
    "flattered_win_inside",
    # B7: strength of the field that beat the horse last time
    "last_race_strength",
    # B8: market drift vs excuses (the over-reaction core)
    "odds_drift_log",
    "pop_drift_pct",
    "excuse_score",
    "overreaction_index",
)


def add_past_run_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add B-group features, aligned to ``frame``'s index."""

    ordered = frame.sort_values(SORT_KEYS, kind="stable").copy()
    by_race = ordered.groupby(RACE_ID)

    # --- post-race facts of each race (only ever exposed via shift(1)) -----
    winner_time = (
        ordered.loc[ordered["finish_position"].eq(1)]
        .groupby(RACE_ID)["race_time_seconds"]
        .min()
    )
    ordered["time_behind_winner"] = ordered["race_time_seconds"] - ordered[RACE_ID].map(winner_time)

    ordered["_dist_band"] = (ordered["distance_m"] // 200.0) * 200.0
    band_keys = ["course_type", "_dist_band"]
    daily = (
        ordered.groupby(band_keys + ["race_date"], observed=True)["race_time_seconds"]
        .median()
        .reset_index(name="_daily_median")
        .sort_values(band_keys + ["race_date"], kind="stable")
    )
    daily["_speed_ref"] = daily.groupby(band_keys, observed=True)["_daily_median"].transform(
        lambda s: s.expanding().median().shift(1)
    )
    ref = daily.set_index(band_keys + ["race_date"])["_speed_ref"]
    ref_key = pd.MultiIndex.from_arrays(
        [ordered["course_type"], ordered["_dist_band"], ordered["race_date"]]
    )
    ordered["speed_resid"] = ordered["race_time_seconds"] - pd.Series(
        ref.reindex(ref_key).to_numpy(), index=ordered.index
    )

    last3f_rank = by_race["last3f"].rank(method="min", ascending=True)
    ordered["last3f_rank_pct"] = last3f_rank / by_race["last3f"].transform("count")
    ordered["_last3f_top2"] = (last3f_rank <= 2).astype(float).where(last3f_rank.notna())

    passing = ordered["通過"].astype("string").str.strip()
    parts = passing.str.split("-")
    runners = ordered["field_size_running"].where(ordered["field_size_running"] > 0)
    # to_numeric on string dtype yields nullable Float64; normalize to numpy float64
    first_corner = pd.to_numeric(parts.str[0], errors="coerce").astype("float64")
    last_corner = pd.to_numeric(parts.str[-1], errors="coerce").astype("float64")
    ordered["early_pos_pct"] = first_corner / runners
    ordered["late_pos_pct"] = last_corner / runners
    ordered["ground_gained"] = ordered["early_pos_pct"] - ordered["late_pos_pct"]

    strength = (
        ordered.loc[ordered["finish_position"].le(3)]
        .groupby(RACE_ID)["market_prob"]
        .mean()
    )
    ordered["race_strength"] = ordered[RACE_ID].map(strength)

    ordered["_horse_no_pct"] = ordered["horse_no"] / runners

    # --- shift(1) per horse: expose only strictly earlier starts -----------
    horse = ordered["horse_name"]
    grouped = ordered.groupby("horse_name", sort=False)

    ordered["last_time_behind_winner"] = grouped["time_behind_winner"].shift(1)
    ordered["best_time_behind_winner_career"] = (
        ordered["last_time_behind_winner"].groupby(horse).cummin()
    )
    ordered["avg_time_behind_winner_3"] = (
        ordered["last_time_behind_winner"]
        .groupby(horse)
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    ordered["last_speed_resid"] = grouped["speed_resid"].shift(1)
    ordered["best_speed_resid_career"] = ordered["last_speed_resid"].groupby(horse).cummin()
    older = pd.concat([grouped["speed_resid"].shift(k) for k in (2, 3, 4)], axis=1)
    ordered["speed_resid_trend"] = ordered["last_speed_resid"] - older.mean(axis=1)

    ordered["last_last3f_rank_pct"] = grouped["last3f_rank_pct"].shift(1)
    ordered["last_last3f_top2"] = grouped["_last3f_top2"].shift(1)

    ordered["last_early_pos_pct"] = grouped["early_pos_pct"].shift(1)
    ordered["last_late_pos_pct"] = grouped["late_pos_pct"].shift(1)
    ordered["last_ground_gained"] = grouped["ground_gained"].shift(1)
    ordered["last_horse_no_pct"] = grouped["_horse_no_pct"].shift(1)

    last_finish = grouped["finish_position"].shift(1)
    closer = (
        ordered["last_last3f_top2"].eq(1.0)
        & ordered["last_early_pos_pct"].ge(0.7)
        & last_finish.ge(4)
    )
    ordered["closer_excuse"] = closer.fillna(False).astype("int8")
    wide_loss = ordered["last_horse_no_pct"].ge(0.75) & last_finish.ge(6)
    ordered["excusable_loss_draw"] = wide_loss.fillna(False).astype("int8")
    flattered = ordered["last_horse_no_pct"].le(0.2) & last_finish.le(2)
    ordered["flattered_win_inside"] = flattered.fillna(False).astype("int8")

    ordered["last_race_strength"] = grouped["race_strength"].shift(1)

    ordered["odds_drift_log"] = ordered["log_win_odds"] - np.log(grouped["win_odds"].shift(1))
    ordered["pop_drift_pct"] = ordered["popularity_pct"] - grouped["popularity_pct"].shift(1)
    ordered["excuse_score"] = (
        ordered["closer_excuse"].astype("int16") + ordered["excusable_loss_draw"].astype("int16")
    )
    # 人気を落とした(pop_drift>0) × 言い訳がある = 群衆の過剰反応ゾーン
    ordered["overreaction_index"] = ordered["pop_drift_pct"] * ordered["excuse_score"]

    features = ordered.loc[:, list(PAST_RUN_FEATURE_COLUMNS)].reindex(frame.index)
    base = frame.drop(columns=list(PAST_RUN_FEATURE_COLUMNS), errors="ignore")
    return pd.concat([base, features], axis=1)
