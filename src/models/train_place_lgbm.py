"""複勝EV walk-forward 検証 (レシピC4: win-place tension の収穫可能性判定)。

複勝プール監査 (results/20260613_place_pool_audit.md) で単勝[1.0,3.0)帯の
複勝ROIが0.83-0.90と判明。損益分岐に必要な条件付きリフトは約+11-20%で、
単勝プール (+25%超、IG≈0で不可能と確定) より大幅に低い。本モジュールは

  Phase A: P(place) を市場アンカーからの残差として学習
           アンカー = logit(q) の2パラメータlogistic較正 (train期間でIRLS推定、
           市場情報のみ使用)。LightGBM binary + init_score=anchor。
           検証binary_loglossで早期停止。
  ゲート : IG_val(zone) = BCE_anchor − BCE_model を賭けゾーン内で測り、
           <= 0 ならその年は0賭け。
  賭け   : EV = P̂(place) × O_fuku_min > 1+τ。
           O_fuku_min は SPEC-2 odds_final の複勝レンジ下限。
           保守的に下限だけを使い、上限は診断列として保存する。
           τ は検証期間 (cacheカバー行のみ) の総利益で選ぶ (τ=∞=0賭け含む)。
  決済   : cache の実複勝払戻 (同着・降着・2着払いを自動反映)。

ラベル: is_place = 着順<=2 (7頭以下) / <=3 (8頭以上)。2015年から全期間利用可。
払戻データは2022年以降のみのため、テストは2023-2026の4フォールド。

実行:
  py -m src.models.train_place_lgbm --smoke   # 2023のみ・低ラウンド
  py -m src.models.train_place_lgbm           # 本走 (2023-2026)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.analysis.place_pool_audit import load_place_payout_map
from src.models.train_value_lgbm import (
    VAL_DATE_FRACTION,
    build_frame,
)

DEFAULT_ZONE_LO, DEFAULT_ZONE_HI = 1.0, 1.4
TAU_GRID = (0.00, 0.02, 0.05, 0.08, 0.12, 0.20, float("inf"))
MIN_VAL_BETS = 30
N_BOOTSTRAP = 10_000
ODDS_FINAL_DIR = Path("scraper/data/odds_final")

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def fit_logit_calibration(x: np.ndarray, y: np.ndarray, *, iters: int = 50) -> np.ndarray:
    """p = sigmoid(a + b*x) の2パラメータIRLS。市場アンカーの推定に使う。"""
    design = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        p = sigmoid(design @ w)
        grad = design.T @ (p - y)
        curv = p * (1.0 - p) + 1e-9
        hess = design.T @ (design * curv[:, None])
        step = np.linalg.solve(hess, grad)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return w


def bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def bootstrap_roi_ci(dates: np.ndarray, payout: np.ndarray,
                     n_boot: int = N_BOOTSTRAP, seed: int = 20260613) -> tuple[float, float]:
    frame = pd.DataFrame({"d": pd.to_datetime(dates), "pay": payout, "n": 1.0})
    by_date = frame.groupby("d").sum()
    pay_d, n_d = by_date["pay"].to_numpy(), by_date["n"].to_numpy()
    if len(pay_d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pay_d), size=(n_boot, len(pay_d)))
    roi = pay_d[idx].sum(axis=1) / (100.0 * np.maximum(n_d[idx].sum(axis=1), 1e-9))
    return (float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)))


def _positive_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def load_place_odds_range_map(
    odds_dir: str | Path = ODDS_FINAL_DIR,
) -> tuple[dict[tuple[str, int], tuple[float, float]], set[str]]:
    """SPEC-2 odds_final から (race_id, 馬番) -> (複勝下限, 上限) を返す。"""
    ranges: dict[tuple[str, int], tuple[float, float]] = {}
    races: set[str] = set()
    for path in sorted(Path(odds_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("api_status") != "result":
                    continue
                race_id = str(record.get("race_id"))
                fuku = record.get("fuku") or {}
                if not fuku:
                    continue
                races.add(race_id)
                for horse_no, values in fuku.items():
                    if not isinstance(values, list) or len(values) < 2:
                        continue
                    lo = _positive_float(values[0])
                    hi = _positive_float(values[1])
                    if lo is None or hi is None:
                        continue
                    ranges[(race_id, int(horse_no))] = (lo, hi)
    return ranges, races


def prepare(feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    df, feature_columns = build_frame(feature_set)
    if "finish_position" not in df.columns:
        raise ValueError("finish_position 列が必要 (build_frame の出力を確認)")
    # 複勝ラベル: 7頭以下は2着払い
    df["is_place"] = np.where(
        df["field_size_running"] <= 7,
        df["finish_position"] <= 2,
        df["finish_position"] <= 3,
    ).astype("int8")

    payout_map, races = load_place_payout_map()
    df["race_id"] = df["race_id"].astype(str)
    in_cache = df["race_id"].isin(races)
    keys = list(zip(df.loc[in_cache, "race_id"], df.loc[in_cache, "horse_no"].astype(int)))
    payouts = pd.Series(np.nan, index=df.index)
    payouts.loc[in_cache] = [float(payout_map.get(k, 0.0)) for k in keys]
    df["place_payout"] = payouts  # NaN = cache未カバー (金額評価から除外)

    odds_map, odds_races = load_place_odds_range_map()
    in_odds = df["race_id"].isin(odds_races)
    odds_keys = list(zip(df.loc[in_odds, "race_id"], df.loc[in_odds, "horse_no"].astype(int)))
    fuku_min = pd.Series(np.nan, index=df.index)
    fuku_max = pd.Series(np.nan, index=df.index)
    ranges = [odds_map.get(k) for k in odds_keys]
    fuku_min.loc[in_odds] = [r[0] if r else np.nan for r in ranges]
    fuku_max.loc[in_odds] = [r[1] if r else np.nan for r in ranges]
    df["fuku_min_odds"] = fuku_min
    df["fuku_max_odds"] = fuku_max

    q = np.exp(df["logq"].to_numpy())
    df["x_logit_q"] = np.log(q) - np.log1p(-q)
    return df, feature_columns


def run_fold(df: pd.DataFrame, feature_columns: list[str], test_year: int,
             *, rounds: int, stopping: int, zone_lo: float, zone_hi: float) -> dict:
    zone = (df["win_odds"] >= zone_lo) & (df["win_odds"] < zone_hi)
    actionable = df["place_payout"].notna() & df["fuku_min_odds"].notna()
    test = df.loc[(df["year"] == test_year) & zone & actionable]
    pre = df.loc[df["year"] < test_year]
    dates = pre["race_date"].sort_values().unique()
    n_val_dates = max(1, int(len(dates) * VAL_DATE_FRACTION))
    val_start = dates[-n_val_dates]
    train = pre.loc[pre["race_date"] < val_start]
    val = pre.loc[pre["race_date"] >= val_start]

    # --- 市場アンカー (train のみで推定) ---
    w = fit_logit_calibration(train["x_logit_q"].to_numpy(), train["is_place"].to_numpy(float))
    anchor = lambda d: w[0] + w[1] * d["x_logit_q"].to_numpy()

    train_ds = lgb.Dataset(train[feature_columns], label=train["is_place"],
                           init_score=anchor(train), free_raw_data=False)
    val_ds = lgb.Dataset(val[feature_columns], label=val["is_place"],
                         init_score=anchor(val), reference=train_ds, free_raw_data=False)
    booster = lgb.train(
        LGB_PARAMS, train_ds, num_boost_round=rounds,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping, verbose=False)],
    )
    best_it = booster.best_iteration or rounds

    def predict(d: pd.DataFrame) -> np.ndarray:
        raw = booster.predict(d[feature_columns], raw_score=True, num_iteration=best_it)
        return sigmoid(anchor(d) + raw)

    # --- ゾーン内 IG ゲート (検証) ---
    val_zone = val.loc[(val["win_odds"] >= zone_lo) & (val["win_odds"] < zone_hi)]
    y_vz = val_zone["is_place"].to_numpy(float)
    ig_val = bce(y_vz, sigmoid(anchor(val_zone))) - bce(y_vz, predict(val_zone))

    result = {
        "test_year": test_year, "n_train": len(train), "n_val": len(val),
        "n_test_zone": len(test), "best_it": int(best_it), "ig_val_zone": float(ig_val),
        "n_val_zone": int(len(val_zone)),
        "n_val_actionable": int(
            val_zone["place_payout"].notna().sum()
            if "fuku_min_odds" not in val_zone
            else (val_zone["place_payout"].notna() & val_zone["fuku_min_odds"].notna()).sum()
        ),
        "anchor": [float(w[0]), float(w[1])],
    }

    if ig_val <= 0.0:
        result.update({"phase": "no-bet", "tau_star": float("inf"), "n_bets": 0,
                       "roi": np.nan, "roi_ci": (np.nan, np.nan), "hit_rate": np.nan,
                       "profit": 0.0, "baseline_zone_roi": float(
                           test["place_payout"].sum() / (100.0 * max(len(test), 1)))})
        result["bets"] = _empty_bets(test, test_year)
        return result

    # --- τ選択 (検証のゾーン内・cacheカバー行・複勝下限ありの総利益最大化) ---
    val_bet = val_zone.loc[val_zone["place_payout"].notna() & val_zone["fuku_min_odds"].notna()]
    ev_val = predict(val_bet) * val_bet["fuku_min_odds"].to_numpy()
    best_tau, best_profit = float("inf"), 0.0
    tau_rows = []
    for tau in TAU_GRID:
        mask = ev_val > 1.0 + tau
        profit = float(val_bet.loc[mask, "place_payout"].sum() - 100.0 * mask.sum())
        tau_rows.append({"tau": tau, "val_bets": int(mask.sum()), "val_profit": profit})
        if mask.sum() >= MIN_VAL_BETS and profit > best_profit:
            best_tau, best_profit = tau, profit
    result["tau_table"] = tau_rows

    p_test = predict(test)
    ev_test = p_test * test["fuku_min_odds"].to_numpy()
    sel = ev_test > 1.0 + best_tau
    bets = test.loc[sel, ["race_id", "race_date", "horse_no", "win_odds",
                          "popularity", "fuku_min_odds", "fuku_max_odds",
                          "is_place", "place_payout"]].copy()
    bets["p_model"] = p_test[sel]
    bets["ev"] = ev_test[sel]
    bets["test_year"] = test_year
    n = len(bets)
    payout_sum = float(bets["place_payout"].sum())
    result.update({
        "phase": "bet" if n else "bet(0)",
        "tau_star": best_tau,
        "n_bets": int(n),
        "roi": payout_sum / (100.0 * n) if n else np.nan,
        "roi_ci": bootstrap_roi_ci(bets["race_date"].to_numpy(),
                                   bets["place_payout"].to_numpy()) if n else (np.nan, np.nan),
        "hit_rate": float((bets["place_payout"] > 0).mean()) if n else np.nan,
        "profit": payout_sum - 100.0 * n,
        "baseline_zone_roi": float(test["place_payout"].sum() / (100.0 * max(len(test), 1))),
        "bets": bets,
    })
    return result


def _empty_bets(test: pd.DataFrame, test_year: int) -> pd.DataFrame:
    columns = [
        "race_id", "race_date", "horse_no", "win_odds", "popularity",
        "fuku_min_odds", "fuku_max_odds", "is_place", "place_payout",
    ]
    bets = test.iloc[0:0][[c for c in columns if c in test.columns]].copy()
    bets["p_model"] = np.array([], dtype=float)
    bets["ev"] = np.array([], dtype=float)
    bets["test_year"] = test_year
    return bets


def format_fold(r: dict) -> str:
    lo, hi = r["roi_ci"]
    roi = f"{r['roi']:.4f}" if r["n_bets"] else "-"
    ci = f"[{lo:.3f}, {hi:.3f}]" if r["n_bets"] else "-"
    hit = f"{r['hit_rate']:.3f}" if r["n_bets"] else "-"
    tau = "inf" if np.isinf(r["tau_star"]) else f"{r['tau_star']:.2f}"
    return (
        f"| {r['test_year']} | {r['phase']} | {r['best_it']} | {r['ig_val_zone']:+.5f} "
        f"| {tau} | {r['n_test_zone']} | {r['n_bets']} | {roi} | {ci} | {hit} "
        f"| {r['profit']:+,.0f} | {r['baseline_zone_roi']:.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--features", choices=("v2", "v3"), default="v3")
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--zone-lo", type=float, default=DEFAULT_ZONE_LO)
    parser.add_argument("--zone-hi", type=float, default=DEFAULT_ZONE_HI)
    args = parser.parse_args()
    if not args.zone_lo < args.zone_hi:
        raise ValueError("--zone-lo must be smaller than --zone-hi")

    rounds, stopping = (150, 30) if args.smoke else (800, 50)
    test_years = [2023] if args.smoke else [2023, 2024, 2025, 2026]
    if args.years:
        test_years = args.years

    df, feature_columns = prepare(args.features)
    print(f"frame: {df.shape}, features: {len(feature_columns)}, "
          f"zone=[{args.zone_lo},{args.zone_hi}), cache行: {int(df['place_payout'].notna().sum()):,}, "
          f"fuku_min行: {int(df['fuku_min_odds'].notna().sum()):,}")
    # ラベル健全性: cacheカバー行で is_place と実払戻>0 の不一致率 (降着等で小さく非ゼロ)
    covered = df.loc[df["place_payout"].notna()]
    mismatch = float((covered["is_place"].astype(bool) != (covered["place_payout"] > 0)).mean())
    print(f"ラベル/実払戻 不一致率 (降着・同着等): {mismatch:.5f}")

    results = [run_fold(df, feature_columns, y, rounds=rounds, stopping=stopping,
                        zone_lo=args.zone_lo, zone_hi=args.zone_hi)
               for y in test_years]

    all_bets = pd.concat([r["bets"] for r in results], ignore_index=True)
    pooled_roi, pooled_ci = np.nan, (np.nan, np.nan)
    if len(all_bets):
        pooled_roi = all_bets["place_payout"].sum() / (100.0 * len(all_bets))
        pooled_ci = bootstrap_roi_ci(all_bets["race_date"].to_numpy(),
                                     all_bets["place_payout"].to_numpy())

    header = (
        "| 年 | phase | best_it | IG_val(zone) | τ* | zone行 | 賭数 | ROI | 95%CI "
        "| 的中率 | 損益 | 全買いROI(zone) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    summary = "\n".join([header] + [format_fold(r) for r in results])
    print(summary)
    print(f"\nプール: 賭数={len(all_bets)}, ROI={pooled_roi:.4f}, CI={pooled_ci}")

    zone_tag = f"z{args.zone_lo:g}_{args.zone_hi:g}".replace(".", "p")
    tag = ("smoke" if args.smoke else "walkforward") + f"_fukumin_{args.features}_{zone_tag}"
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(out_dir / f"place_lgbm_bets_{tag}.csv", index=False)
    payload = {
        "summary_table": summary,
        "pooled": {"n_bets": int(len(all_bets)), "roi": float(pooled_roi),
                   "ci": list(pooled_ci)},
        "folds": [{k: v for k, v in r.items() if k != "bets"} for r in results],
        "params": {"rounds": rounds, "stopping": stopping,
                   "zone": [args.zone_lo, args.zone_hi],
                   "ev_source": "p_model * fuku_min_odds",
                   "tau_grid": TAU_GRID, "min_val_bets": MIN_VAL_BETS,
                   "lgb": LGB_PARAMS, "features": args.features,
                   "label_mismatch_rate": mismatch},
    }
    (out_dir / f"place_lgbm_metrics_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"成果物: {out_dir}/place_lgbm_bets_{tag}.csv, place_lgbm_metrics_{tag}.json")


if __name__ == "__main__":
    main()
