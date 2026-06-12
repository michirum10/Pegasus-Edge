"""価値最大化 LightGBM の walk-forward 学習・バックテスト (設計書 §8-§9)。

- 市場アンカー: p = softmax(log q + F)。F は LightGBM の生スコア
- 学習: ValueObjective (CEウォームアップ → 価値項アニーリング)
- τ* は訓練末尾の検証期間で対数富最大化により選び、テスト年に固定適用
- ベースライン: 全馬フラット / 1番人気フラット (ハーネス健全性検査)

実行:
  py -m src.models.train_value_lgbm --smoke   # 配管確認 (2017年テスト, 80round)
  py -m src.models.train_value_lgbm           # 本走 (2021-2026 walk-forward)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.dataloader import (
    BASE_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    load_dataset,
)
from src.models.value_objective import (
    ValueObjective,
    kelly_stakes,
    race_groups,
    softmax_by_race,
)

CATEGORICAL_COLUMNS = (
    "sex",
    "course_type",
    "jockey",
    "trainer",
    "owner",
    "horse_last_course_type",
)

# float("inf") = 「その年は一切賭けない」選択肢。検証対数富が全τで負なら
# argmax は inf (対数富 0.0) を選び、運用は0賭けになる
TAU_GRID = (0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.50, float("inf"))
KAPPA = 0.10
BETA_END = 8.0
VAL_DATE_FRACTION = 0.15
N_BOOTSTRAP = 10_000

LGB_PARAMS = {
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


def build_frame() -> tuple[pd.DataFrame, list[str]]:
    """学習・検証に使う1枚のフレームを返す (レース単位で行が連続)。"""
    ds = load_dataset(include_history=True)
    df = pd.concat([ds.X, ds.y, ds.metadata[["race_id", "race_date", "horse_name"]]], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)

    # 勝者がちょうど1頭のレースのみ残す (同着・勝者行欠落の除外)
    n_win = df.groupby("race_id")["is_win"].transform("sum")
    dropped_races = df.loc[n_win != 1, "race_id"].nunique()
    df = df.loc[n_win == 1].reset_index(drop=True)
    print(f"勝者が1頭でないため除外したレース: {dropped_races}")

    # 市場確率 q はフィルタ後の実出走行で再規格化する
    inv = 1.0 / df["win_odds"].to_numpy()
    sum_inv = df.assign(_inv=inv).groupby("race_id")["_inv"].transform("sum").to_numpy()
    df["logq"] = np.log(inv / sum_inv)
    df["year"] = pd.to_datetime(df["race_date"]).dt.year

    feature_columns = [c for c in (*BASE_FEATURE_COLUMNS, *HISTORY_FEATURE_COLUMNS)
                       if c in df.columns]
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df, feature_columns


def make_objective(df: pd.DataFrame, *, warmup: int, total: int, tau: float) -> ValueObjective:
    starts, sizes = race_groups(df["race_id"].to_numpy())
    return ValueObjective(
        logq=df["logq"].to_numpy(dtype=np.float64),
        odds=df["win_odds"].to_numpy(dtype=np.float64),
        y=df["is_win"].to_numpy(dtype=np.float64),
        starts=starts,
        sizes=sizes,
        kappa=KAPPA,
        tau=tau,
        beta_end=BETA_END,
        warmup_rounds=warmup,
        total_rounds=total,
    )


def predict_probs(booster: lgb.Booster, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    raw = booster.predict(df[feature_columns], num_iteration=booster.num_trees())
    starts, sizes = race_groups(df["race_id"].to_numpy())
    return softmax_by_race(df["logq"].to_numpy() + raw, starts, sizes)


def flat_backtest(df: pd.DataFrame, select: np.ndarray) -> dict:
    """選択行に単勝100円フラット買い。"""
    sel = df.loc[select]
    n = len(sel)
    if n == 0:
        return {"n_bets": 0, "roi": np.nan, "hit_rate": np.nan,
                "profit": 0.0, "max_drawdown": np.nan, "roi_ci": (np.nan, np.nan)}
    payout = sel["win_payout_per_100yen"].to_numpy()
    profit = payout - 100.0
    cum = np.cumsum(profit)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    ci = bootstrap_roi_ci(sel["race_date"].to_numpy(), payout, np.full(n, 100.0))
    return {
        "n_bets": int(n),
        "roi": float(payout.sum() / (100.0 * n)),
        "hit_rate": float(sel["is_win"].mean()),
        "profit": float(profit.sum()),
        "max_drawdown": float((peak - cum).max()),
        "roi_ci": ci,
    }


def kelly_backtest(df: pd.DataFrame, p: np.ndarray, tau: float) -> dict:
    """レース毎リバランスの fractional Kelly。対数富成長を測る。"""
    f = kelly_stakes(p, df["win_odds"].to_numpy(), kappa=KAPPA, beta=BETA_END, tau=tau)
    c = df["is_win"].to_numpy() * df["win_odds"].to_numpy() - 1.0
    starts, _ = race_groups(df["race_id"].to_numpy())
    w = 1.0 + np.add.reduceat(f * c, starts)
    staked = np.add.reduceat(f, starts)
    bet_races = staked > 1e-4
    log_growth = float(np.sum(np.log(np.maximum(w[bet_races], 1e-9))))
    curve = np.exp(np.cumsum(np.log(np.maximum(w, 1e-9))))
    peak = np.maximum.accumulate(curve)
    return {
        "n_bet_races": int(bet_races.sum()),
        "log_growth": log_growth,
        "final_bankroll": float(curve[-1]) if len(curve) else 1.0,
        "max_drawdown_pct": float((1.0 - curve / peak).max()) if len(curve) else np.nan,
    }


def bootstrap_roi_ci(dates: np.ndarray, payout: np.ndarray, stake: np.ndarray,
                     n_boot: int = N_BOOTSTRAP, seed: int = 20260612) -> tuple[float, float]:
    """開催日クラスタブートストラップによる ROI 95%CI。"""
    frame = pd.DataFrame({"d": pd.to_datetime(dates), "pay": payout, "stk": stake})
    by_date = frame.groupby("d").sum()
    pay_d, stk_d = by_date["pay"].to_numpy(), by_date["stk"].to_numpy()
    if len(pay_d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pay_d), size=(n_boot, len(pay_d)))
    roi = pay_d[idx].sum(axis=1) / np.maximum(stk_d[idx].sum(axis=1), 1e-9)
    return (float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)))


def tune_tau(df: pd.DataFrame, p: np.ndarray) -> tuple[float, pd.DataFrame]:
    """検証期間の Kelly 対数富を最大にする τ を選ぶ。"""
    rows = []
    for tau in TAU_GRID:
        k = kelly_backtest(df, p, tau)
        e = p * df["win_odds"].to_numpy() - 1.0
        rows.append({"tau": tau, "val_log_growth": k["log_growth"],
                     "val_bet_races": k["n_bet_races"],
                     "val_flat_bets": int((e > tau).sum())})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_log_growth"].idxmax(), "tau"]
    return float(best), table


def run_fold(df: pd.DataFrame, feature_columns: list[str], test_year: int,
             *, rounds: int, warmup: int) -> dict:
    test = df.loc[df["year"] == test_year]
    pre = df.loc[df["year"] < test_year]
    dates = pre["race_date"].sort_values().unique()
    n_val_dates = max(1, int(len(dates) * VAL_DATE_FRACTION))
    val_start = dates[-n_val_dates]
    train = pre.loc[pre["race_date"] < val_start]
    val = pre.loc[pre["race_date"] >= val_start]

    obj = make_objective(train, warmup=warmup, total=rounds, tau=0.08)
    dataset = lgb.Dataset(train[feature_columns], label=train["is_win"],
                          free_raw_data=False)
    params = dict(LGB_PARAMS, objective=obj)
    booster = lgb.train(params, dataset, num_boost_round=rounds)

    p_val = predict_probs(booster, val, feature_columns)
    p_test = predict_probs(booster, test, feature_columns)

    # 較正の健全性: モデルCEが市場CEを上回る(悪化する)なら情報を壊している
    ce_market = float(-test.loc[test["is_win"] == 1, "logq"].mean())
    winners = test["is_win"].to_numpy() == 1
    ce_model = float(-np.log(np.maximum(p_test[winners], 1e-300)).mean())

    tau_star, tau_table = tune_tau(val, p_val)
    e_test = p_test * test["win_odds"].to_numpy() - 1.0
    flat = flat_backtest(test, e_test > tau_star)
    kelly = kelly_backtest(test, p_test, tau_star)

    bets = test.loc[e_test > tau_star,
                    ["race_id", "race_date", "horse_no", "horse_name",
                     "win_odds", "popularity", "is_win", "win_payout_per_100yen"]].copy()
    bets["edge"] = e_test[e_test > tau_star]
    bets["p_model"] = p_test[e_test > tau_star]
    bets["test_year"] = test_year

    baseline_all = flat_backtest(test, np.ones(len(test), dtype=bool))
    baseline_fav = flat_backtest(test, test["popularity"].to_numpy() == 1)

    return {
        "test_year": test_year,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "tau_star": tau_star,
        "tau_table": tau_table,
        "ce_market": ce_market, "ce_model": ce_model,
        "flat": flat, "kelly": kelly,
        "baseline_all": baseline_all, "baseline_fav": baseline_fav,
        "bets": bets,
    }


def format_fold(r: dict) -> str:
    flat, kelly = r["flat"], r["kelly"]
    lo, hi = flat["roi_ci"]
    return (
        f"| {r['test_year']} | {r['n_train']:,} | {r['tau_star']:.2f} "
        f"| {r['ce_market']:.4f} | {r['ce_model']:.4f} "
        f"| {flat['n_bets']} | {flat['roi'] if flat['n_bets'] else float('nan'):.4f} "
        f"| [{lo:.3f}, {hi:.3f}] | {flat['hit_rate']:.3f} "
        f"| {kelly['log_growth']:+.4f} | {r['baseline_all']['roi']:.4f} "
        f"| {r['baseline_fav']['roi']:.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="配管確認の小規模実行")
    parser.add_argument("--pure-ce", action="store_true",
                        help="価値項を無効化 (warmup=rounds)。較正力の診断用")
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="テスト年を限定 (例: --years 2025)")
    args = parser.parse_args()

    rounds, warmup = (80, 40) if args.smoke else (500, 150)
    if args.pure_ce:
        warmup = rounds
    test_years = [2017] if args.smoke else [2021, 2022, 2023, 2024, 2025, 2026]
    if args.years:
        test_years = args.years

    df, feature_columns = build_frame()
    print(f"frame: {df.shape}, features: {len(feature_columns)}")

    results = [run_fold(df, feature_columns, y, rounds=rounds, warmup=warmup)
               for y in test_years]

    all_bets = pd.concat([r["bets"] for r in results], ignore_index=True)
    pooled_ci = (np.nan, np.nan)
    pooled_roi = np.nan
    if len(all_bets):
        pooled_roi = all_bets["win_payout_per_100yen"].sum() / (100.0 * len(all_bets))
        pooled_ci = bootstrap_roi_ci(all_bets["race_date"].to_numpy(),
                                     all_bets["win_payout_per_100yen"].to_numpy(),
                                     np.full(len(all_bets), 100.0))

    header = (
        "| 年 | train行 | τ* | CE市場 | CEモデル | 賭数 | ROI(flat) | 95%CI "
        "| 的中率 | Kelly対数富 | 全馬ROI | 1人気ROI |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header] + [format_fold(r) for r in results]
    summary = "\n".join(lines)
    print(summary)
    print(f"\nプール: 賭数={len(all_bets)}, ROI={pooled_roi:.4f}, CI={pooled_ci}")

    tag = "smoke" if args.smoke else ("pure_ce" if args.pure_ce else "walkforward")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(out_dir / f"value_lgbm_bets_{tag}.csv", index=False)

    payload = {
        "summary_table": summary,
        "pooled": {"n_bets": int(len(all_bets)), "roi": float(pooled_roi),
                   "ci": list(pooled_ci)},
        "tau_tables": {str(r["test_year"]): r["tau_table"].to_dict("records")
                       for r in results},
        "params": {"rounds": rounds, "warmup": warmup, "kappa": KAPPA,
                   "beta_end": BETA_END, "tau_grid": TAU_GRID, "lgb": LGB_PARAMS},
    }
    (out_dir / f"value_lgbm_metrics_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"成果物: {out_dir}/value_lgbm_bets_{tag}.csv, value_lgbm_metrics_{tag}.json")


if __name__ == "__main__":
    main()
