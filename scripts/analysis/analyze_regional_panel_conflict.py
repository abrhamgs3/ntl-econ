"""
Panel regressions with conflict controls (ACLED events and fatalities).

Reads merged panel with conflict: data/tabular/regional_ntl_conflict_panel.csv
Runs full sample and exclusion (Addis Ababa, Dire Dawa, Harari) with entity + time FE.
"""
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.paths import TABULAR_DIR  # noqa: E402

INPUT = TABULAR_DIR / "regional_ntl_conflict_panel.csv"
SPECIAL = ["Addis Ababa", "Dire Dawa", "Harari"]


def load_panel(path=INPUT):
    df = pd.read_csv(path)
    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["lnNTL", "Leader_region", "events", "fatalities"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lnNTL", "Leader_region", "year", "ADM1_EN"]).copy()
    df = df.set_index(["ADM1_EN", "year"])
    return df


def run_fe(df):
    exog = sm.add_constant(df[["Leader_region", "events", "fatalities"]])
    mod = PanelOLS(df["lnNTL"], exog, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered")
    return res


def main():
    df = load_panel()
    res_full = run_fe(df)
    print("\nFull sample (entity + year FE, clustered)")
    print(res_full.summary)

    df_excl = df.loc[~df.index.get_level_values(0).isin(SPECIAL)]
    res_excl = run_fe(df_excl)
    print("\nExcluding Addis/Dire/Harari (entity + year FE, clustered)")
    print(res_excl.summary)


if __name__ == "__main__":
    main()
