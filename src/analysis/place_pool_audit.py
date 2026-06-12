"""複勝プールの歪み監査 (レシピ群C・C4 win-place tension の前段)。

単勝市場は現4資産に対し効率的と確定した (v2/v3 walk-forward)。複勝プールは
流動性が低く計算コストの高い券種であるため、同じ歪み (FLB) がより大きく
残っている可能性がある。`_payout_cache.jsonl` の実複勝払戻 (2022-2026、
約14,000レース) を使い、全馬複勝100円フラット買いの素ROIを帯別×年別に
実測し、同一レース集合での単勝ROIとの較差 (tension) を地図化する。

的中判定は着順ではなく cache の fukusho エントリ自体を使う (同着・降着・
7頭以下の2着払いを自動的に正しく扱うため)。

実行:
  py -m src.analysis.place_pool_audit
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataloader import load_dataset

ODDS_BANDS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf"))
POPULARITY_BANDS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 99))
N_BOOTSTRAP = 10_000


def load_place_payout_map(cache_path: str | Path = "_payout_cache.jsonl") -> tuple[dict, set]:
    """(race_id, 馬番) -> 複勝払戻円 の写像と、複勝データを持つrace_id集合を返す。"""
    payout: dict[tuple[str, int], int] = {}
    races: set[str] = set()
    with Path(cache_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            entries = record.get("fukusho")
            if not entries:
                continue
            race_id = str(record["race_id"])
            races.add(race_id)
            for entry in entries:
                for num in entry["nums"]:
                    payout[(race_id, int(num))] = int(entry["yen"])
    return payout, races


def build_frame() -> pd.DataFrame:
    ds = load_dataset(include_history=False)
    df = pd.concat([ds.X, ds.y, ds.metadata[["race_id", "race_date"]]], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = pd.to_datetime(df["race_date"]).dt.year
    return df


def attach_place_payout(df: pd.DataFrame, payout_map: dict, races: set) -> pd.DataFrame:
    sub = df.loc[df["race_id"].isin(races)].copy()
    keys = list(zip(sub["race_id"], sub["horse_no"].astype(int)))
    sub["place_payout_per_100yen"] = [float(payout_map.get(k, 0)) for k in keys]
    sub["is_place_hit"] = (sub["place_payout_per_100yen"] > 0).astype("int8")
    return sub


def band_table(df: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    grouped = df.assign(_band=key).groupby(["year", "_band"], observed=True)
    out = grouped.agg(
        n=("is_place_hit", "size"),
        place_hit=("is_place_hit", "mean"),
        place_payout=("place_payout_per_100yen", "sum"),
        win_payout=("win_payout_per_100yen", "sum"),
    ).reset_index()
    out["place_roi"] = out["place_payout"] / (out["n"] * 100.0)
    out["win_roi"] = out["win_payout"] / (out["n"] * 100.0)
    return out.rename(columns={"_band": label})


def pivot_summary(table: pd.DataFrame, label: str) -> pd.DataFrame:
    pivot = table.pivot(index=label, columns="year", values="place_roi")
    total = table.groupby(label, observed=True).agg(
        n=("n", "sum"),
        place_payout=("place_payout", "sum"),
        win_payout=("win_payout", "sum"),
        hits=("place_hit", lambda s: np.nan),  # placeholder, recomputed below
    )
    pivot["place_ALL"] = total["place_payout"] / (total["n"] * 100.0)
    pivot["win_ALL"] = total["win_payout"] / (total["n"] * 100.0)
    pivot["tension"] = pivot["place_ALL"] - pivot["win_ALL"]
    pivot["n_total"] = total["n"]
    return pivot.round(4)


def hit_decomposition(df: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    grouped = df.assign(_band=key).groupby("_band", observed=True)
    out = grouped.agg(
        n=("is_place_hit", "size"),
        hit_rate=("is_place_hit", "mean"),
        payout_sum=("place_payout_per_100yen", "sum"),
        hits=("is_place_hit", "sum"),
    )
    out["avg_payout_when_hit"] = out["payout_sum"] / out["hits"].replace(0, np.nan)
    out["place_roi"] = out["payout_sum"] / (out["n"] * 100.0)
    out.index.name = label
    return out[["n", "hit_rate", "avg_payout_when_hit", "place_roi"]].round(4)


def bootstrap_roi_ci(dates: np.ndarray, payout: np.ndarray,
                     n_boot: int = N_BOOTSTRAP, seed: int = 20260613) -> tuple[float, float]:
    """開催日クラスタブートストラップ (フラット100円賭け前提) の ROI 95%CI。"""
    frame = pd.DataFrame({"d": pd.to_datetime(dates), "pay": payout, "n": 1.0})
    by_date = frame.groupby("d").sum()
    pay_d, n_d = by_date["pay"].to_numpy(), by_date["n"].to_numpy()
    if len(pay_d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pay_d), size=(n_boot, len(pay_d)))
    roi = pay_d[idx].sum(axis=1) / (100.0 * np.maximum(n_d[idx].sum(axis=1), 1e-9))
    return (float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)))


def main() -> None:
    payout_map, races = load_place_payout_map()
    df = build_frame()
    sub = attach_place_payout(df, payout_map, races)

    # 健全性: cache側の複勝的中頭数とresults側の行の突合
    hits_per_race = sub.groupby("race_id")["is_place_hit"].sum()
    matched_races = int((hits_per_race > 0).sum())
    row_keys = set(zip(sub["race_id"], sub["horse_no"].astype(int)))
    unmatched_payouts = sum(1 for k in payout_map if k not in row_keys)

    overall_place = sub["place_payout_per_100yen"].sum() / (len(sub) * 100.0)
    overall_win = sub["win_payout_per_100yen"].sum() / (len(sub) * 100.0)
    ci_place = bootstrap_roi_ci(sub["race_date"].to_numpy(),
                                sub["place_payout_per_100yen"].to_numpy())

    odds_band = pd.cut(sub["win_odds"], bins=ODDS_BANDS, right=False)
    odds_pivot = pivot_summary(band_table(sub, odds_band, "odds_band"), "odds_band")
    odds_decomp = hit_decomposition(sub, odds_band, "odds_band")

    pop_labels = pd.Series(pd.NA, index=sub.index, dtype="string")
    for lo, hi in POPULARITY_BANDS:
        mask = sub["popularity"].between(lo, hi)
        pop_labels[mask] = f"{lo}-{hi}人気" if lo != hi else f"{lo}人気"
    pop_pivot = pivot_summary(band_table(sub, pop_labels, "pop_band"), "pop_band")

    size_labels = pd.Series("8頭以上(3着払い)", index=sub.index, dtype="string")
    size_labels[sub["field_size_running"] <= 7] = "7頭以下(2着払い)"
    size_pivot = pivot_summary(band_table(sub, size_labels, "field_class"), "field_class")

    # 帯別CI: ALLで tension が最大の帯について日次クラスタCIを出す
    best_band = odds_pivot["tension"].idxmax()
    best_mask = odds_band == best_band
    ci_best = bootstrap_roi_ci(sub.loc[best_mask, "race_date"].to_numpy(),
                               sub.loc[best_mask, "place_payout_per_100yen"].to_numpy())

    today = dt.date.today()
    lines = [
        f"# {today:%Y%m%d} 複勝プール歪み監査（レシピ群C前段・win-place tension）",
        "",
        f"- 日付: {today:%Y-%m-%d}",
        "- 目的: 単勝市場効率性の確定 (v3) を受け、低流動性の複勝プールに搾取可能な歪みが残るかを実払戻で実測する。",
        "- 方法: payout cache の fukusho 実払戻を的中判定に用い、全出走馬に複勝100円フラット買いした素ROIを帯別×年別に集計。同一レース集合の単勝ROIとの差を tension とする。",
        "",
        "## 対象データ",
        "",
        f"- 複勝払戻を持つレース: {len(races):,}（cache全体）",
        f"- results と突合できた行: {len(sub):,}行 / {sub['race_id'].nunique():,}レース（的中行あり: {matched_races:,}レース）",
        f"- cache側払戻エントリのうち results 行に対応しない数: {unmatched_payouts:,}",
        f"- 1レースあたり複勝的中頭数の分布: {hits_per_race.value_counts().sort_index().to_dict()}",
        "",
        "## 全体",
        "",
        f"- 複勝フラットROI: **{overall_place:.4f}**（95%CI [{ci_place[0]:.4f}, {ci_place[1]:.4f}]、日次クラスタブートストラップ）",
        f"- 同一レース集合の単勝フラットROI: {overall_win:.4f}",
        f"- 全体 tension（複勝−単勝）: {overall_place - overall_win:+.4f}",
        "",
        "## オッズ帯別（単勝オッズで層化）",
        "",
        odds_pivot.to_markdown(),
        "",
        "### 的中率×平均払戻への分解",
        "",
        odds_decomp.to_markdown(),
        "",
        f"- tension最大帯: **{best_band}** の複勝ROI 95%CI [{ci_best[0]:.4f}, {ci_best[1]:.4f}]",
        "",
        "## 人気帯別",
        "",
        pop_pivot.to_markdown(),
        "",
        "## 頭数クラス別（2着払い/3着払い）",
        "",
        size_pivot.to_markdown(),
        "",
        "## 解釈メモ",
        "",
        "- place_ALL が 1.0 を超える帯があれば無条件エッジ（要CI確認・要年別安定性）。",
        "- 1.0 未満でも tension が大きく正の帯は「単勝で買うより複勝で買うべき層」であり、条件付きモデル（C4: win-place tension 特徴量）の優先ゾーン。",
        "- 複勝の的中判定・払戻は cache 実測値であり同着・降着・2着払いを自動反映している。",
        "- 対象は2022-2026のみ（2024年は約1,100レース欠落）。2015-2021は複勝払戻が存在しないため検証不能。",
    ]
    out_path = Path(f"results/{today:%Y%m%d}_place_pool_audit.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"レポート保存: {out_path}")
    print(f"複勝フラットROI: {overall_place:.4f} CI{ci_place} / 単勝: {overall_win:.4f}")
    print(odds_pivot.to_string())


if __name__ == "__main__":
    main()
