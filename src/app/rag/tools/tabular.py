from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


def summarize_table(path: Path, sheet: Optional[str] = None, head: int = 5) -> Dict[str, Any]:
    df = load_table(path, sheet)
    preview = df.head(head).to_dict(orient="records")
    summary = df.describe(include="all").fillna("").to_dict()
    return {"preview": preview, "summary": summary}


def aggregate_column(
    path: Path, column: str, agg: str = "sum", sheet: Optional[str] = None
) -> Dict[str, Any]:
    df = load_table(path, sheet)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {path}")
    series = df[column].dropna()
    if agg == "sum":
        value = series.sum()
    elif agg == "mean":
        value = series.mean()
    elif agg == "max":
        value = series.max()
    elif agg == "min":
        value = series.min()
    else:
        raise ValueError(f"Unsupported aggregation '{agg}'")
    return {"column": column, "aggregation": agg, "value": float(value)}
