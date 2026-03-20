"""
Regression analysis functions for regional panel (panel regressions, summaries, etc).
"""
from linearmodels.panel import BetweenOLS, PanelOLS, PooledOLS, RandomEffects

def add_lagged_leader_region(df_panel):
    df_panel = df_panel.copy()
    df_panel['Leader_region_lag'] = df_panel.groupby(level=0)['Leader_region'].shift(1)
    return df_panel

def run_panel_regressions(df_panel):
    """
    Run pooled, between, and within (fixed effects) panel regressions.
    Returns a dict of regression results.
    """
    df_panel = add_lagged_leader_region(df_panel)
    reg_table = {
        "(1) Pooled": PooledOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag", data=df_panel.dropna(subset=["Leader_region_lag"])
        ).fit(cov_type="clustered"),
        "(2) Pooled + Entity Effects": PanelOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag + EntityEffects",
            data=df_panel.dropna(subset=["Leader_region_lag"]),
        ).fit(cov_type="clustered"),
        "(3) Pooled + Time Effects": PanelOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag + TimeEffects",
            data=df_panel.dropna(subset=["Leader_region_lag"]),
        ).fit(cov_type="clustered"),
        "(4) Within (Entity + Time FE)": PanelOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag + EntityEffects + TimeEffects",
            data=df_panel.dropna(subset=["Leader_region_lag"]),
        ).fit(cov_type="clustered"),
        "(5) Between": BetweenOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag", data=df_panel.dropna(subset=["Leader_region_lag"])
        ).fit(cov_type="clustered"),
        "(6) Random Effects": RandomEffects.from_formula(
            formula="lnNTL ~ 1 + Leader_region_lag", data=df_panel.dropna(subset=["Leader_region_lag"])
        ).fit(cov_type="clustered"),
    }
    return reg_table

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    import sys

    # Adjust path to import from scripts.common.paths
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.common.paths import TABULAR_DIR

    # Update: Use data/tabular folder for input
    INPUT_PATH = Path(ROOT) / "data" / "tabular" / "regional_ntl_panel_data.csv"

    def prepare_panel(df):
        df_panel = df.dropna(subset=["lnNTL", "Leader_region", "ADM1_EN", "year"]).copy()
        df_panel = df_panel.set_index(["ADM1_EN", "year"])
        return df_panel

    df = pd.read_csv(INPUT_PATH)
    df_panel = prepare_panel(df)
    reg_table = run_panel_regressions(df_panel)
    for name, res in reg_table.items():
        print(f"\n{name}\n{'-'*len(name)}\n{res.summary}")
