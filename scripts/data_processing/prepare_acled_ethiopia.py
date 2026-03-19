"""
Filter ACLED CSV to Ethiopia and produce national and ADM1 yearly aggregates.

Inputs:
- data/tabular/ACLED Data_2026-03-17.csv  (or update ACLED_PATH below)

Outputs:
- data/tabular/acled_ethiopia_yearly.csv
- data/tabular/acled_ethiopia_admin1_yearly.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tabular"
ACLED_PATH = DATA_DIR / "ACLED Data_2026-03-17.csv"
OUT_YEARLY = DATA_DIR / "acled_ethiopia_yearly.csv"
OUT_ADMIN1 = DATA_DIR / "acled_ethiopia_admin1_yearly.csv"


def chunk_filter():
    usecols = [
        "event_date",
        "year",
        "country",
        "admin1",
        "fatalities",
        "event_type",
        "sub_event_type",
    ]
    dtype = {
        "country": "category",
        "admin1": "category",
        "event_type": "category",
        "sub_event_type": "category",
    }
    chunks = []
    for chunk in pd.read_csv(
        ACLED_PATH,
        usecols=usecols,
        dtype=dtype,
        sep=";",
        parse_dates=["event_date"],
        chunksize=250_000,
        low_memory=False,
    ):
        eth = chunk[chunk["country"] == "Ethiopia"].copy()
        if eth.empty:
            continue
        eth["fatalities"] = pd.to_numeric(eth["fatalities"], errors="coerce").fillna(0)
        chunks.append(eth)
    if not chunks:
        raise ValueError("No Ethiopia rows found in ACLED file.")
    return pd.concat(chunks, ignore_index=True)


def aggregate_yearly(df):
    out = (
        df.groupby("year")
        .agg(
            events=("event_type", "size"),
            fatalities=("fatalities", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )
    return out


def aggregate_admin1_yearly(df):
    out = (
        df.groupby(["admin1", "year"])
        .agg(
            events=("event_type", "size"),
            fatalities=("fatalities", "sum"),
        )
        .reset_index()
        .rename(columns={"admin1": "ADM1_EN"})
        .sort_values(["ADM1_EN", "year"])
    )
    return out


def main():
    df = chunk_filter()
    yearly = aggregate_yearly(df)
    admin1 = aggregate_admin1_yearly(df)

    yearly.to_csv(OUT_YEARLY, index=False)
    admin1.to_csv(OUT_ADMIN1, index=False)

    print(f"Wrote {OUT_YEARLY} ({len(yearly)} rows)")
    print(f"Wrote {OUT_ADMIN1} ({len(admin1)} rows)")


if __name__ == "__main__":
    main()
