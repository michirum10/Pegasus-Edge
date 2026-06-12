"""Brand overbet features (recipe A4): jockey/trainer crowd premium.

Famous riders and stables attract market money beyond their true lift.  For
each brand key we track, over strictly earlier dates (same-day excluded via
the trailing encoder), the decayed gap between the market's implied
probability and the realized win rate of that brand's mounts:

    overbet = E_hist[market_prob - is_win]   (shrunk toward 0)

Positive = the crowd systematically overprices this brand (fade signal);
negative = underprices (follow signal).  A zone-corrected ROI with shrinkage
toward the trailing global ROI is exposed alongside.
"""

from __future__ import annotations

import pandas as pd

from src.features.trailing_stats import add_trailing_group_stats

BRAND_KEYS: tuple[str, ...] = ("jockey", "trainer")

BRAND_FEATURE_COLUMNS: tuple[str, ...] = (
    "jockey_overbet",
    "jockey_roi",
    "jockey_n_eff",
    "trainer_overbet",
    "trainer_roi",
    "trainer_n_eff",
)


def add_brand_overbet_features(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...] = BRAND_KEYS,
    half_life_days: float = 730.0,
    shrinkage_k: float = 200.0,
) -> pd.DataFrame:
    """Add A4 features.  Contract as in ``add_trailing_group_stats``: rows must
    already be filtered to valid outcomes."""

    out = add_trailing_group_stats(frame, [], prefix="_bglob", half_life_days=half_life_days)
    helper_prefixes = ["_bglob_hist_"]
    for key in keys:
        out = add_trailing_group_stats(out, [key], prefix=f"_{key}", half_life_days=half_life_days)
        helper_prefixes.append(f"_{key}_hist_")
        n = out[f"_{key}_hist_n"]
        gap = out[f"_{key}_hist_q_mean"] - out[f"_{key}_hist_win_rate"]
        out[f"{key}_overbet"] = (n * gap / (n + shrinkage_k)).where(n > 0, 0.0)
        roi = (n * out[f"_{key}_hist_roi"] + shrinkage_k * out["_bglob_hist_roi"]) / (
            n + shrinkage_k
        )
        out[f"{key}_roi"] = roi.where(n > 0, out["_bglob_hist_roi"])
        out[f"{key}_n_eff"] = n.fillna(0.0)

    helpers = [c for c in out.columns if any(c.startswith(p) for p in helper_prefixes)]
    return out.drop(columns=helpers)
