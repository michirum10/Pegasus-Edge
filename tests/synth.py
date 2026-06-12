"""Synthetic raw-frame builders mirroring the race_results/race_meta contract."""

from __future__ import annotations

import pandas as pd

from src.data.dataloader import merge_results_with_meta, prepare_model_frame


def result_row(
    race_id: str,
    horse_no: int,
    *,
    odds: float | None,
    popularity: float | None,
    finish: object,
    name: str | None = None,
    time: str = "1:36.5",
    agari: str = "35.0",
    body_weight: str = "480(+2)",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "race_id": race_id,
        "着順": finish,
        "枠番": (horse_no + 1) // 2,
        "馬番": horse_no,
        "馬名": name or f"馬_{race_id}_{horse_no}",
        "性齢": "牡4",
        "斤量": 56.0,
        "騎手": "騎手A",
        "タイム": time,
        "着差": "",
        "通過": "3-3",
        "上り": agari,
        "単勝": odds,
        "人気": popularity,
        "馬体重": body_weight,
        "調教師": "調教師A",
        "馬主": "馬主A",
        "賞金(万円)": "",
    }
    row.update(overrides)
    return row


def meta_row(
    race_id: str,
    kaisai_date: str,
    *,
    course_type: str = "turf",
    distance_m: int = 1600,
) -> dict[str, object]:
    return {
        "race_id": race_id,
        "kaisai_date": kaisai_date,
        "course_type": course_type,
        "distance_m": distance_m,
    }


def prepared_frame(
    result_rows: list[dict[str, object]],
    meta_rows: list[dict[str, object]],
    *,
    include_history: bool = False,
) -> pd.DataFrame:
    merged = merge_results_with_meta(pd.DataFrame(result_rows), pd.DataFrame(meta_rows))
    return prepare_model_frame(merged, include_history=include_history)
