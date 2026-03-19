import pandas as pd
from linearmodels.panel import BetweenOLS, PanelOLS, PooledOLS

from scripts.common.paths import REGIONAL_RESULTS_DIR, TABULAR_DIR

INPUT_PATH = TABULAR_DIR / "regional_ntl_panel_data.csv"
RESULTS_DIR = REGIONAL_RESULTS_DIR
SUMMARY_PATH = RESULTS_DIR / "regional_panel_excluding_special_summary.txt"
EXCLUDED_REGIONS = ["Addis Ababa", "Dire Dawa", "Harari"]


def load_and_clean_data(input_path=INPUT_PATH, excluded_regions=EXCLUDED_REGIONS):
    """
    Load the active regional panel and exclude special city-regions.
    """
    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^\s*$")]
    df = df.loc[:, ~df.columns.duplicated()]

    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])

    required_columns = ["ADM1_EN", "year", "Leader_region", "lnNTL"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df = df[~df["ADM1_EN"].isin(excluded_regions)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["Leader_region"] = pd.to_numeric(df["Leader_region"], errors="coerce")
    df["lnNTL"] = pd.to_numeric(df["lnNTL"], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    df["year"] = df["year"].astype(int)

    return df.sort_values(["ADM1_EN", "year"]).reset_index(drop=True)


def prepare_panel(df):
    """
    Convert the cleaned frame to a panel index used by linearmodels.
    """
    return df.set_index(["ADM1_EN", "year"])


def run_panel_regressions(df_panel):
    """
    Estimate pooled, between, and within panel regressions.
    """
    return {
        "(1) Pooled": PooledOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region", data=df_panel
        ).fit(cov_type="clustered"),
        "(2) Between": BetweenOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region", data=df_panel
        ).fit(cov_type="clustered"),
        "(3) Within": PanelOLS.from_formula(
            formula="lnNTL ~ 1 + Leader_region + EntityEffects", data=df_panel
        ).fit(cov_type="clustered"),
    }


def save_results(results, output_path=SUMMARY_PATH):
    """
    Save all regression summaries to a single text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for name, result in results.items():
            file.write(f"{name}\n")
            file.write(f"{'-' * len(name)}\n")
            file.write(result.summary.as_text())
            file.write("\n\n")

    print(f"Saved panel summaries to: {output_path}")


def main():
    df = load_and_clean_data()
    print("Columns after cleaning and exclusion:")
    print(df.columns.tolist())
    print(df.head())

    df_panel = prepare_panel(df)
    results = run_panel_regressions(df_panel)

    print("\n--- Panel Data Regression Results (Excluding Special Regions) ---\n")
    for name, result in results.items():
        print(f"\n{name}\n{'-' * len(name)}\n{result.summary}")

    save_results(results)


if __name__ == "__main__":
    main()
