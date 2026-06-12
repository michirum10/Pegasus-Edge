"""価値最大化 LightGBM の walk-forward 学習・バックテスト v2 (設計書 §8-§9)。

v2 の構造 (v1 の「エッジなし」診断を受けた IG ゲート方式):

  Phase A: 純CEで学習し、検証CEで早期停止。
           IG_val := CE市場(検証) - CEモデル(検証) を測る。
  ゲート : IG_val <= 0 なら市場較正に勝てていないため、その年は0賭け。
           (いかなるステーキング層も IG<=0 のモデルでは数理的に勝てない)
  Phase B: IG_val > 0 のときだけ、価値項 (Soft-Kelly対数富) を
           アニーリングしながら継続学習し、τ* を検証期間で選んで賭ける。

特徴量 v2: dataloader の基本+馬履歴に加え、騎手・調教師の expanding 成績と
racecourse / track_condition (processed_races.csv) を使用。

実行:
  py -m src.models.train_value_lgbm --smoke   # 配管確認 (2017年テスト)
  py -m src.models.train_value_lgbm           # 本走 (2021-2026 walk-forward)
  py -m src.models.train_value_lgbm --pure-ce # Phase B を行わない診断モード
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
    assert_no_feature_leakage,
    load_dataset,
    load_prepared_frame,
)
from src.data.extra_features import (
    ACTOR_FEATURE_COLUMNS,
    RACE_CONTEXT_COLUMNS,
    add_actor_history,
    add_race_context,
)
from src.features.brand import BRAND_FEATURE_COLUMNS, add_brand_overbet_features
from src.features.market import MARKET_FEATURE_COLUMNS, add_market_microstructure_features
from src.features.past_run import PAST_RUN_FEATURE_COLUMNS, add_past_run_features
from src.features.trailing_stats import CALIBRATION_FEATURE_COLUMNS, add_odds_zone_calibration
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
    "racecourse",
    "track_condition",
)

# float("inf") = 「その年は一切賭けない」。検証対数富が全τで負なら選ばれる
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
    "metric": "none",
    "verbose": -1,
    "num_threads": 0,
}

PHASE_A_MODEL_PATH = Path("data/processed/_phase_a_model.txt")


def build_frame(feature_set: str = "v2") -> tuple[pd.DataFrame, list[str]]:
    """学習・検証に使う1枚のフレームを返す (レース単位で行が連続)。

    v2: 基本+馬履歴+騎手/調教師expanding+場・馬場 (45特徴量)
    v3: v2 + レシピA1-A4/B群 = 市場ミクロ構造・FLBゾーン較正・過去走ロールアップ
        (勝ち馬タイム差/スピード残差/上がり順位/通過/過剰反応指数)・ブランド過剰
        人気指数。設計は results/20260612_roi_feature_recipes.md
    """
    if feature_set == "v3":
        frame = load_prepared_frame(include_history=True)
        df = frame.loc[
            frame["finish_position"].notna()
            & frame["win_odds"].notna()
            & frame["popularity"].notna()
        ].copy()
    else:
        ds = load_dataset(include_history=True)
        df = pd.concat([ds.X, ds.y, ds.metadata[["race_id", "race_date", "horse_name"]]], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)

    # 勝者がちょうど1頭のレースのみ残す (同着・勝者行欠落の除外)
    n_win = df.groupby("race_id")["is_win"].transform("sum")
    dropped_races = df.loc[n_win != 1, "race_id"].nunique()
    df = df.loc[n_win == 1].reset_index(drop=True)
    print(f"勝者が1頭でないため除外したレース: {dropped_races}")

    df = add_race_context(df)
    df = add_actor_history(df)

    extra_columns: list[str] = []
    if feature_set == "v3":
        df = add_market_microstructure_features(df)
        df = add_odds_zone_calibration(df)
        df = add_past_run_features(df)
        df = add_brand_overbet_features(df)
        extra_columns = [
            *MARKET_FEATURE_COLUMNS,
            *CALIBRATION_FEATURE_COLUMNS,
            *PAST_RUN_FEATURE_COLUMNS,
            *BRAND_FEATURE_COLUMNS,
        ]

    # 市場確率 q はフィルタ後の実出走行で再規格化する
    inv = 1.0 / df["win_odds"].to_numpy()
    sum_inv = df.assign(_inv=inv).groupby("race_id")["_inv"].transform("sum").to_numpy()
    df["logq"] = np.log(inv / sum_inv)
    df["year"] = pd.to_datetime(df["race_date"]).dt.year

    feature_columns = [
        c
        for c in (
            *BASE_FEATURE_COLUMNS,
            *HISTORY_FEATURE_COLUMNS,
            *ACTOR_FEATURE_COLUMNS,
            *RACE_CONTEXT_COLUMNS,
            *extra_columns,
        )
        if c in df.columns
    ]
    assert_no_feature_leakage(df[feature_columns])
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


def make_val_ce_feval(val: pd.DataFrame):
    """検証レース内softmax CE を返す feval (早期停止用)。"""
    logq = val["logq"].to_numpy()
    starts, sizes = race_groups(val["race_id"].to_numpy())
    winners = val["is_win"].to_numpy() == 1

    def feval(preds: np.ndarray, _eval_data) -> tuple[str, float, bool]:
        p = softmax_by_race(logq + preds, starts, sizes)
        ce = float(-np.log(np.maximum(p[winners], 1e-300)).mean())
        return "val_ce", ce, False

    return feval


def predict_probs(
    booster: lgb.Booster,
    df: pd.DataFrame,
    feature_columns: list[str],
    num_iteration: int | None = None,
) -> np.ndarray:
    raw = booster.predict(df[feature_columns], num_iteration=num_iteration)
    starts, sizes = race_groups(df["race_id"].to_numpy())
    return softmax_by_race(df["logq"].to_numpy() + raw, starts, sizes)


def race_ce(df: pd.DataFrame, p: np.ndarray) -> float:
    winners = df["is_win"].to_numpy() == 1
    return float(-np.log(np.maximum(p[winners], 1e-300)).mean())


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
    """検証期間の Kelly 対数富を最大にする τ を選ぶ (τ=inf は0賭け=対数富0)。"""
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


def _empty_bets(test: pd.DataFrame, test_year: int) -> pd.DataFrame:
    bets = test.iloc[0:0][["race_id", "race_date", "horse_no", "horse_name",
                           "win_odds", "popularity", "is_win", "win_payout_per_100yen"]].copy()
    bets["edge"] = np.array([], dtype=float)
    bets["p_model"] = np.array([], dtype=float)
    bets["test_year"] = test_year
    return bets


def run_fold(df: pd.DataFrame, feature_columns: list[str], test_year: int,
             *, phase_a_rounds: int, stopping: int, phase_b_rounds: int,
             pure_ce: bool, no_early_stop: bool = False,
             no_ig_gate: bool = False) -> dict:
    test = df.loc[df["year"] == test_year]
    pre = df.loc[df["year"] < test_year]
    dates = pre["race_date"].sort_values().unique()
    n_val_dates = max(1, int(len(dates) * VAL_DATE_FRACTION))
    val_start = dates[-n_val_dates]
    train = pre.loc[pre["race_date"] < val_start]
    val = pre.loc[pre["race_date"] >= val_start]

    # --- Phase A: 純CE + 検証CE早期停止 ---
    ce_obj = make_objective(train, warmup=10 ** 9, total=10 ** 9, tau=0.08)
    train_ds = lgb.Dataset(train[feature_columns], label=train["is_win"],
                           free_raw_data=False)
    val_ds = lgb.Dataset(val[feature_columns], label=val["is_win"],
                         reference=train_ds, free_raw_data=False)
    callbacks = ([] if no_early_stop
                 else [lgb.early_stopping(stopping_rounds=stopping, verbose=False)])
    booster = lgb.train(
        dict(LGB_PARAMS, objective=ce_obj),
        train_ds,
        num_boost_round=phase_a_rounds,
        valid_sets=[val_ds],
        valid_names=["val"],
        feval=make_val_ce_feval(val),
        callbacks=callbacks,
    )
    best_it = booster.best_iteration or phase_a_rounds

    p_val = predict_probs(booster, val, feature_columns, num_iteration=best_it)
    ce_market_val = float(-val.loc[val["is_win"] == 1, "logq"].mean())
    ig_val = ce_market_val - race_ce(val, p_val)

    # --- IG ゲート (--no-ig-gate で無効化、診断用) ---
    if ig_val <= 0.0 and not no_ig_gate:
        phase = "A:no-bet"
        p_test = predict_probs(booster, test, feature_columns, num_iteration=best_it)
        tau_star, tau_table = float("inf"), tune_tau(val, p_val)[1]
        flat = flat_backtest(test, np.zeros(len(test), dtype=bool))
        kelly = {"n_bet_races": 0, "log_growth": 0.0,
                 "final_bankroll": 1.0, "max_drawdown_pct": 0.0}
        bets = _empty_bets(test, test_year)
    else:
        if pure_ce:
            phase = "A:bet"
            final_booster, num_it = booster, best_it
        else:
            # --- Phase B: best_it から価値項アニーリングで継続学習 ---
            phase = "B:value"
            PHASE_A_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(PHASE_A_MODEL_PATH), num_iteration=best_it)
            value_obj = make_objective(train, warmup=0, total=phase_b_rounds, tau=0.08)
            final_booster = lgb.train(
                dict(LGB_PARAMS, objective=value_obj, learning_rate=0.02),
                train_ds,
                num_boost_round=phase_b_rounds,
                init_model=str(PHASE_A_MODEL_PATH),
            )
            num_it = None
        p_val = predict_probs(final_booster, val, feature_columns, num_iteration=num_it)
        p_test = predict_probs(final_booster, test, feature_columns, num_iteration=num_it)
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

    ce_market_test = float(-test.loc[test["is_win"] == 1, "logq"].mean())
    ce_model_test = race_ce(test, p_test)

    return {
        "test_year": test_year,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "phase": phase, "best_it": int(best_it), "ig_val": float(ig_val),
        "tau_star": tau_star,
        "tau_table": tau_table,
        "ce_market": ce_market_test, "ce_model": ce_model_test,
        "flat": flat, "kelly": kelly,
        "baseline_all": flat_backtest(test, np.ones(len(test), dtype=bool)),
        "baseline_fav": flat_backtest(test, test["popularity"].to_numpy() == 1),
        "bets": bets,
    }


def format_fold(r: dict) -> str:
    flat, kelly = r["flat"], r["kelly"]
    lo, hi = flat["roi_ci"]
    roi = f"{flat['roi']:.4f}" if flat["n_bets"] else "-"
    ci = f"[{lo:.3f}, {hi:.3f}]" if flat["n_bets"] else "-"
    hit = f"{flat['hit_rate']:.3f}" if flat["n_bets"] else "-"
    return (
        f"| {r['test_year']} | {r['phase']} | {r['best_it']} | {r['ig_val']:+.4f} "
        f"| {r['ce_market']:.4f} | {r['ce_model']:.4f} | {r['tau_star']:.2f} "
        f"| {flat['n_bets']} | {roi} | {ci} | {hit} "
        f"| {kelly['log_growth']:+.4f} | {r['baseline_all']['roi']:.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="配管確認の小規模実行")
    parser.add_argument("--pure-ce", action="store_true",
                        help="Phase B (価値項) を行わない診断モード")
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="テスト年を限定 (例: --years 2025)")
    parser.add_argument("--features", choices=("v2", "v3"), default="v2",
                        help="v3 = レシピA1-A4/B群を追加 (過去走ロールアップ等)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="診断用: Phase A の検証CE早期停止を無効化 (v1旧挙動の近似)")
    parser.add_argument("--no-ig-gate", action="store_true",
                        help="診断用: IGゲートを無効化し IG<=0 でも賭け評価に進む")
    args = parser.parse_args()

    if args.smoke:
        phase_a_rounds, stopping, phase_b_rounds = 150, 30, 60
        test_years = [2017]
    else:
        phase_a_rounds, stopping, phase_b_rounds = 800, 50, 200
        test_years = [2021, 2022, 2023, 2024, 2025, 2026]
    if args.years:
        test_years = args.years

    df, feature_columns = build_frame(args.features)
    print(f"frame: {df.shape}, features: {len(feature_columns)} ({args.features})")

    results = [
        run_fold(df, feature_columns, y, phase_a_rounds=phase_a_rounds,
                 stopping=stopping, phase_b_rounds=phase_b_rounds,
                 pure_ce=args.pure_ce, no_early_stop=args.no_early_stop,
                 no_ig_gate=args.no_ig_gate)
        for y in test_years
    ]

    all_bets = pd.concat([r["bets"] for r in results], ignore_index=True)
    pooled_roi, pooled_ci = np.nan, (np.nan, np.nan)
    if len(all_bets):
        pooled_roi = all_bets["win_payout_per_100yen"].sum() / (100.0 * len(all_bets))
        pooled_ci = bootstrap_roi_ci(all_bets["race_date"].to_numpy(),
                                     all_bets["win_payout_per_100yen"].to_numpy(),
                                     np.full(len(all_bets), 100.0))

    header = (
        "| 年 | phase | best_it | IG_val | CE市場 | CEモデル | τ* | 賭数 "
        "| ROI(flat) | 95%CI | 的中率 | Kelly対数富 | 全馬ROI |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    summary = "\n".join([header] + [format_fold(r) for r in results])
    print(summary)
    print(f"\nプール: 賭数={len(all_bets)}, ROI={pooled_roi:.4f}, CI={pooled_ci}")

    base_tag = "smoke" if args.smoke else ("pure_ce" if args.pure_ce else "walkforward")
    tag = f"{base_tag}_{args.features}"
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(out_dir / f"value_lgbm_bets_{tag}.csv", index=False)

    payload = {
        "summary_table": summary,
        "pooled": {"n_bets": int(len(all_bets)), "roi": float(pooled_roi),
                   "ci": list(pooled_ci)},
        "folds": [{k: v for k, v in r.items() if k not in ("bets", "tau_table")}
                  for r in results],
        "params": {"phase_a_rounds": phase_a_rounds, "stopping": stopping,
                   "phase_b_rounds": phase_b_rounds, "kappa": KAPPA,
                   "beta_end": BETA_END, "tau_grid": TAU_GRID, "lgb": LGB_PARAMS},
    }
    (out_dir / f"value_lgbm_metrics_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"成果物: {out_dir}/value_lgbm_bets_{tag}.csv, value_lgbm_metrics_{tag}.json")


if __name__ == "__main__":
    main()
