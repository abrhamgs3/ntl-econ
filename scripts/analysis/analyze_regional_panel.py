import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
from scripts.analysis.plot_regional_panel import plot_gini
from scripts.analysis.plot_regional_panel import plot_ntl_histogram, plot_leader_vs_nonleader_trends, plot_ntl_boxplot_by_leader
from scripts.analysis.regression_regional_panel import run_panel_regressions

from scripts.common.paths import REGIONAL_FIGURES_DIR, REGIONAL_RESULTS_DIR, TABULAR_DIR

INPUT_PATH = TABULAR_DIR / "regional_ntl_panel_data.csv"
RESULTS_DIR = REGIONAL_RESULTS_DIR
FIGURES_DIR = REGIONAL_FIGURES_DIR
GINI_OUTPUT_PATH = RESULTS_DIR / "regional_ntl_gini_by_year.csv"
GINI_FIGURE_PATH = FIGURES_DIR / "regional_ntl_gini_trend.png"
REG_SUMMARY_FULL = RESULTS_DIR / "regional_panel_full_summary.txt"
REG_SUMMARY_EXCL = RESULTS_DIR / "regional_panel_excluding_special_summary.txt"

SUMMARY_STATS_PATH = RESULTS_DIR / "regional_panel_summary_stats.csv"
NTL_HIST_FIGURE_PATH = FIGURES_DIR / "ntl_histogram.png"
LEADER_TREND_FIGURE_PATH = FIGURES_DIR / "ntl_leader_vs_nonleader_trend.png"
NTL_BOX_FIGURE_PATH = FIGURES_DIR / "ntl_boxplot_by_leader.png"


def gini(array):
    """
    Compute the Gini coefficient for a one-dimensional array.

    The function shifts negative values if needed, adds a small constant to
    avoid division-by-zero issues, and returns a value in [0, 1] for standard
    non-negative inputs.
    """
    values = np.asarray(array, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        raise ValueError("Gini cannot be computed on an empty array.")

    if np.amin(values) < 0:
        values = values - np.amin(values)

    values = np.sort(values + 1e-9)
    n = values.shape[0]
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * values) / (n * np.sum(values))


def load_and_clean_data(input_path=INPUT_PATH):
    """
    Load the active regional NTL panel and keep the columns needed for Gini and regressions.
    """
    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^\s*$")]
    df = df.loc[:, ~df.columns.duplicated()]

    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])

    required_columns = ["ADM1_EN", "year", "NTL", "lnNTL", "Leader_region"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["NTL"] = pd.to_numeric(df["NTL"], errors="coerce")
    df["lnNTL"] = pd.to_numeric(df["lnNTL"], errors="coerce")
    df["Leader_region"] = pd.to_numeric(df["Leader_region"], errors="coerce")
    df = df.dropna(subset=["ADM1_EN", "year", "NTL"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["year", "ADM1_EN"]).reset_index(drop=True)

    return df


def prepare_panel(df):
    """
    Set panel index and drop rows missing regression fields.
    """
    df_panel = df.dropna(subset=["lnNTL", "Leader_region", "ADM1_EN", "year"]).copy()
    df_panel = df_panel.set_index(["ADM1_EN", "year"])
    return df_panel


def compute_gini_by_year(df, value_column="NTL"):
    """
    Compute the yearly Gini series for a given numeric column.
    """
    gini_by_year = (
        df.groupby("year", sort=True)[value_column]
        .apply(gini)
        .reset_index(name="gini")
    )
    return gini_by_year


def save_gini_by_year(gini_by_year, output_path=GINI_OUTPUT_PATH):
    """
    Save the yearly Gini series to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gini_by_year.to_csv(output_path, index=False)
    print(f"Saved yearly Gini coefficients to: {output_path}")





def main(show_plot=True):
    df = load_and_clean_data()
    print("Cleaned regional panel preview:")
    print(df.head())

    # --- Summary statistics ---
    # Include Mean_NTL if present
    stat_cols = [col for col in ["NTL", "lnNTL", "Leader_region", "Mean_NTL"] if col in df.columns]
    summary_stats = df[stat_cols].describe().T
    SUMMARY_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_stats.to_csv(SUMMARY_STATS_PATH)
    print(f"Saved summary statistics to: {SUMMARY_STATS_PATH}")
    print("\nSummary statistics:\n", summary_stats)

    # Run panel regressions (full sample)
    df_panel = prepare_panel(df)
    print("\n--- Panel Data Regression Results (Full Sample) ---\n")
    reg_table = run_panel_regressions(df_panel)
    REG_SUMMARY_FULL.parent.mkdir(parents=True, exist_ok=True)
    with open(REG_SUMMARY_FULL, "w", encoding="utf-8") as f:
        for name, res in reg_table.items():
            print(f"\n{name}\n{'-'*len(name)}\n{res.summary}", file=f)
            print(f"\n{name}\n{'-'*len(name)}\n{res.summary}")

    # Run panel regressions excluding special city-regions
    special = ["Addis Ababa", "Dire Dawa", "Harari"]
    df_excl = df[~df["ADM1_EN"].isin(special)].copy()
    df_panel_excl = prepare_panel(df_excl)
    print("\n--- Panel Data Regression Results (Excluding Addis Ababa, Dire Dawa, Harari) ---\n")
    reg_table_excl = run_panel_regressions(df_panel_excl)
    with open(REG_SUMMARY_EXCL, "w", encoding="utf-8") as f:
        for name, res in reg_table_excl.items():
            print(f"\n{name}\n{'-'*len(name)}\n{res.summary}", file=f)
            print(f"\n{name}\n{'-'*len(name)}\n{res.summary}")

    gini_by_year = compute_gini_by_year(df)
    print("\nGini coefficients by year:")
    print(gini_by_year.to_string(index=False))

    save_gini_by_year(gini_by_year)

    plot_gini(gini_by_year, GINI_FIGURE_PATH, show_plot=show_plot)

    # --- New plots ---
    plot_ntl_histogram(df, NTL_HIST_FIGURE_PATH, show_plot=show_plot)
    plot_leader_vs_nonleader_trends(df, LEADER_TREND_FIGURE_PATH, show_plot=show_plot)
    plot_ntl_boxplot_by_leader(df, NTL_BOX_FIGURE_PATH, show_plot=show_plot)


if __name__ == "__main__":
    main()
