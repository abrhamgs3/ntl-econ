"""
Merge regional NTL panel with ACLED conflict aggregates.

Inputs:
- data/tabular/regional_ntl_panel_data.csv
- data/tabular/acled_ethiopia_admin1_yearly.csv

Output:
- data/tabular/regional_ntl_conflict_panel.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABULAR = ROOT / "data" / "tabular"

NTL_PATH = TABULAR / "regional_ntl_panel_data.csv"
ACLED_ADMIN1 = TABULAR / "acled_ethiopia_admin1_yearly.csv"
OUT_PATH = TABULAR / "regional_ntl_conflict_panel.csv"


def main():
    ntl = pd.read_csv(NTL_PATH)
    acled = pd.read_csv(ACLED_ADMIN1)

    # Clean column names
    # Avoid duplicate year columns: if both exist, drop 'Year' after copying to 'year'
    if "Year" in ntl.columns and "year" in ntl.columns:
        ntl = ntl.drop(columns=["Year"])
    elif "Year" in ntl.columns:
        ntl = ntl.rename(columns={"Year": "year"})
    acled = acled.rename(columns={"year": "year", "ADM1_EN": "ADM1_EN"})

    # Ensure types
    ntl["year"] = pd.to_numeric(ntl["year"], errors="coerce")
    acled["year"] = pd.to_numeric(acled["year"], errors="coerce")

    # Merge
    merged = ntl.merge(
        acled,
        on=["ADM1_EN", "year"],
        how="left",
        suffixes=("", "_acled"),
    )

    # Fill missing conflict metrics with zeros
    for col in ["events", "fatalities"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote merged panel with conflict to: {OUT_PATH} (rows={len(merged)})")


if __name__ == "__main__":
    main()
