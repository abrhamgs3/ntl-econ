"""
Panel FE regressions with distributed lags of leader-region status.

Outputs:
- outputs/results/regional/leader_lag_fe_full.txt
- outputs/results/regional/leader_lag_fe_excluding_special.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.paths import REGIONAL_RESULTS_DIR, TABULAR_DIR  # noqa: E402

INPUT_PATH = TABULAR_DIR / "regional_ntl_panel_data.csv"
OUT_FULL = REGIONAL_RESULTS_DIR / "leader_lag_fe_full.txt"
OUT_EXCL = REGIONAL_RESULTS_DIR / "leader_lag_fe_excluding_special.txt"

SPECIAL = ["Addis Ababa", "Dire Dawa", "Harari"]


def load_panel(path=INPUT_PATH):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^\s*$")]
    df = df.loc[:, ~df.columns.duplicated()]
    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])
    needed = ["ADM1_EN", "year", "lnNTL", "Leader_region"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["lnNTL"] = pd.to_numeric(df["lnNTL"], errors="coerce")
    df["Leader_region"] = pd.to_numeric(df["Leader_region"], errors="coerce")
    df = df.dropna(subset=needed).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["ADM1_EN", "year"])
    # lags by region
    df["Leader_lag1"] = df.groupby("ADM1_EN")["Leader_region"].shift(1)
    df["Leader_lag2"] = df.groupby("ADM1_EN")["Leader_region"].shift(2)
    return df


def run_fe(df, path):
    df_panel = df.dropna(subset=["Leader_region", "Leader_lag1", "Leader_lag2", "lnNTL"]).copy()
    df_panel = df_panel.set_index(["ADM1_EN", "year"])
    exog = sm.add_constant(df_panel[["Leader_region", "Leader_lag1", "Leader_lag2"]])
    mod = PanelOLS(df_panel["lnNTL"], exog, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(res.summary.as_text())
    print(f"Wrote {path}")
    return res


def main():
    df = load_panel()
    res_full = run_fe(df, OUT_FULL)
    df_excl = df[~df["ADM1_EN"].isin(SPECIAL)].copy()
    res_excl = run_fe(df_excl, OUT_EXCL)

    print("\nFull-sample coefficients (entity + year FE, clustered):")
    print(res_full.params)
    print("\nExclude city-regions coefficients:")
    print(res_excl.params)


if __name__ == "__main__":
    main()
