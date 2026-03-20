import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.common.paths import NATIONAL_FIGURES_DIR, NATIONAL_RESULTS_DIR, TABULAR_DIR

INPUT_PATH = TABULAR_DIR / "ethiopia_ntl_data.xlsx"
FIGURES_DIR = NATIONAL_FIGURES_DIR
RESULTS_DIR = NATIONAL_RESULTS_DIR

SCATTER_PATH = FIGURES_DIR / "ntl_vs_log_gdp.png"
NTL_TREND_PATH = FIGURES_DIR / "ntl_trend.png"
GDP_TREND_PATH = FIGURES_DIR / "gdp_trend.png"
COMBINED_TREND_PATH = FIGURES_DIR / "ntl_gdp_combined_trend.png"
SUMMARY_PATH = RESULTS_DIR / "ntl_gdp_regression_summary.txt"
ELASTICITY_PATH = RESULTS_DIR / "ntl_gdp_elasticity_summary.txt"


def load_data(input_path=INPUT_PATH):
    """
    Load the national NTL panel and validate the required columns.
    """
    df = pd.read_excel(input_path)

    required_columns = ["Year", "Mean_NTL", "GDPPC"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mean_NTL"] = pd.to_numeric(df["Mean_NTL"], errors="coerce")
    df["GDPPC_con"] = pd.to_numeric(df["GDPPC"], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    return df.sort_values("Year").reset_index(drop=True)


def prepare_regression_sample(df):
    """
    Restrict to positive observations required for the log-log regression.
    """
    reg_df = df[(df["Mean_NTL"] > 0) & (df["GDPPC_con"] > 0)].copy()

    if reg_df.empty:
        raise ValueError("No valid positive observations remain after regression filtering.")

    reg_df["log_GDP"] = np.log(reg_df["GDPPC_con"])
    reg_df["log_NTL"] = np.log(reg_df["Mean_NTL"])
    return reg_df.reset_index(drop=True)


def ensure_output_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_scatter(reg_df, output_path=SCATTER_PATH, show_plot=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(reg_df["log_GDP"], reg_df["log_NTL"])
    ax.set_xlabel("log(GDP per capita)")
    ax.set_ylabel("log(Mean NTL)")
    ax.set_title("Ethiopia: log(Mean NTL) vs log(GDP per capita)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved scatter plot to: {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_trends(df, show_plot=True):
    start_year = int(df["Year"].min())
    end_year = int(df["Year"].max())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Year"], df["Mean_NTL"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean NTL")
    ax.set_title(f"Ethiopia: Mean NTL Trend ({start_year}-{end_year})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(NTL_TREND_PATH, dpi=300)
    print(f"Saved NTL trend plot to: {NTL_TREND_PATH}")
    if show_plot:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Year"], np.log(df["GDPPC_con"]), marker="o", color="green")
    ax.set_xlabel("Year")
    ax.set_ylabel("log(GDP per capita)")
    ax.set_title(f"Ethiopia: log(GDP per capita) Trend ({start_year}-{end_year})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GDP_TREND_PATH, dpi=300)
    print(f"Saved GDP trend plot to: {GDP_TREND_PATH}")
    if show_plot:
        plt.show()
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean NTL", color="tab:blue")
    ax1.plot(df["Year"], df["Mean_NTL"], marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("log(GDP per capita)", color="tab:green")
    ax2.plot(df["Year"], np.log(df["GDPPC_con"]), marker="s", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title(f"Ethiopia: Mean NTL and log(GDP per capita) Trends ({start_year}-{end_year})")
    fig.tight_layout()
    fig.savefig(COMBINED_TREND_PATH, dpi=300)
    print(f"Saved combined trend plot to: {COMBINED_TREND_PATH}")
    if show_plot:
        plt.show()
    plt.close(fig)


def run_regression(df):
    """
    Estimate the national log-log NTL-GDP relationship.
    """
    X = sm.add_constant(df["log_GDP"])
    model = sm.OLS(df["log_NTL"], X).fit()
    correlation = df[["Mean_NTL", "log_GDP"]].corr().iloc[0, 1]
    elasticity = model.params["log_GDP"]
    return model, correlation, elasticity


def save_regression_outputs(model, correlation, elasticity):
    interpretation = (
        f"Correlation between Mean_NTL and log(GDP): {correlation:.3f}\n"
        f"Elasticity (log-log regression slope): {elasticity:.3f}\n"
        f"Interpretation: A 1% increase in GDP per capita is associated with "
        f"a {elasticity:.2f}% change in NTL.\n"
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(interpretation)
        file.write("\n--- OLS Regression Results ---\n")
        file.write(model.summary().as_text())

    with open(ELASTICITY_PATH, "w", encoding="utf-8") as file:
        file.write(
            f"Elasticity (log-log regression slope): {elasticity:.3f}\n"
            f"Interpretation: A 1% increase in GDP per capita is associated with "
            f"a {elasticity:.2f}% change in NTL.\n"
        )

    print(f"Saved regression summary to: {SUMMARY_PATH}")
    print(f"Saved elasticity summary to: {ELASTICITY_PATH}")


def main(show_plot=True):
    ensure_output_dirs()
    df = load_data()
    reg_df = prepare_regression_sample(df)
    print("National data preview:")
    print(df.head())
    print("\nRegression sample preview:")
    print(reg_df.head())

    plot_scatter(reg_df, show_plot=show_plot)
    plot_trends(df, show_plot=show_plot)

    model, correlation, elasticity = run_regression(reg_df)
    print(f"Correlation between Mean_NTL and log(GDP): {correlation:.3f}")
    print(f"Elasticity (log-log regression slope): {elasticity:.3f}")
    print(model.summary())

    save_regression_outputs(model, correlation, elasticity)


if __name__ == "__main__":
    main()
