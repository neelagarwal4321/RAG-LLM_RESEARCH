from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def run_sql(path: Path, query: str, sheet: Optional[str] = None) -> Dict[str, Any]:
    df = _load_dataframe(path, sheet)
    with sqlite3.connect(":memory:") as conn:
        df.to_sql("data", conn, index=False, if_exists="replace")
        cursor = conn.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {"rows": rows, "columns": columns, "count": len(rows)}


def _load_dataframe(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)
