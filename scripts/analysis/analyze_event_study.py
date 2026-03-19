import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.common.paths import REGIONAL_FIGURES_DIR, REGIONAL_RESULTS_DIR, TABULAR_DIR

INPUT_PATH = TABULAR_DIR / "regional_ntl_panel_data.csv"
RESULTS_DIR = REGIONAL_RESULTS_DIR
FIGURES_DIR = REGIONAL_FIGURES_DIR
SUMMARY_PATH = RESULTS_DIR / "event_study_summary.txt"
COEFFICIENTS_PATH = RESULTS_DIR / "event_study_coefficients.csv"
FIGURE_PATH = FIGURES_DIR / "event_study_plot.png"

EVENT_WINDOW = range(-5, 6)
REFERENCE_PERIOD = -1

# Regions treated as leader birth-regions in the current project setup.
LEADER_PERIODS = {
    "Tigray": (1992, 2012),
    "SNNP": (2012, 2018),
    "Sidama": (2012, 2018),
    "South West Ethiopia": (2012, 2018),
    "Oromia": (2018, 2024),
}


def load_data(input_path=INPUT_PATH):
    """
    Load the active regional panel and standardize the columns needed below.
    """
    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^\s*$")]
    df = df.loc[:, ~df.columns.duplicated()]

    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])

    required_columns = ["ADM1_EN", "year", "lnNTL"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["lnNTL"] = pd.to_numeric(df["lnNTL"], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    df["year"] = df["year"].astype(int)

    return df.sort_values(["ADM1_EN", "year"]).reset_index(drop=True)


def assign_treatment_timing(df):
    """
    Assign leader start/end years and event time for ever-treated regions.

    Event time is defined for all years of ever-treated regions so that lead
    coefficients are genuine pre-treatment comparisons whenever the sample
    contains pre-treatment observations.
    """
    df = df.copy()
    df["Leader_start"] = df["ADM1_EN"].map(lambda region: LEADER_PERIODS.get(region, (np.nan, np.nan))[0])
    df["Leader_end"] = df["ADM1_EN"].map(lambda region: LEADER_PERIODS.get(region, (np.nan, np.nan))[1])
    df["Ever_treated"] = df["Leader_start"].notna().astype(int)
    df["Treated"] = (
        df["Ever_treated"].eq(1)
        & df["year"].between(df["Leader_start"], df["Leader_end"], inclusive="both")
    ).astype(int)
    df["EventTime"] = np.where(
        df["Ever_treated"].eq(1),
        df["year"] - df["Leader_start"],
        np.nan,
    )
    return df


def build_event_dummies(df, event_window=EVENT_WINDOW, reference_period=REFERENCE_PERIOD):
    """
    Create event-time dummies for the requested window.
    """
    df = df.copy()
    event_terms = []

    for k in event_window:
        if k == reference_period:
            continue
        column = f"event_m{abs(k)}" if k < 0 else f"event{k}"
        df[column] = ((df["EventTime"] == k) & df["Ever_treated"].eq(1)).astype(int)
        event_terms.append(column)

    return df, event_terms


def fit_event_study(df, event_terms):
    """
    Estimate an event-study with region and year fixed effects.
    """
    formula = "lnNTL ~ " + " + ".join(event_terms) + " + C(ADM1_EN) + C(year)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["ADM1_EN"]},
    )
    return model


def extract_event_coefficients(model, event_window=EVENT_WINDOW, reference_period=REFERENCE_PERIOD):
    """
    Return a tidy table of event-study coefficients and confidence intervals.
    """
    rows = []
    for k in event_window:
        if k == reference_period:
            continue

        term = f"event_m{abs(k)}" if k < 0 else f"event{k}"
        coef = model.params.get(term, np.nan)
        se = model.bse.get(term, np.nan)
        rows.append(
            {
                "event_time": k,
                "term": term,
                "coef": coef,
                "std_error": se,
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "p_value": model.pvalues.get(term, np.nan),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(model, coefficients, summary_path=SUMMARY_PATH, coefficients_path=COEFFICIENTS_PATH):
    """
    Save the model summary and the event coefficient table.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(coefficients_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(model.summary().as_text())

    print(f"Saved event-study summary to: {summary_path}")
    print(f"Saved event-study coefficients to: {coefficients_path}")


def plot_event_study(coefficients, output_path=FIGURE_PATH, show_plot=True):
    """
    Plot event-study coefficients with 95 percent confidence intervals.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        coefficients["event_time"],
        coefficients["coef"],
        yerr=1.96 * coefficients["std_error"],
        fmt="o",
        capsize=4,
    )
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.axhline(0, color="black", linestyle=":", linewidth=1)
    plt.xlabel("Years Relative to Leader Start")
    plt.ylabel("Effect on ln(NTL)")
    plt.title("Event Study: Leader Birth-Region Effect")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved event-study plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main(show_plot=True):
    df = load_data()
    df = assign_treatment_timing(df)
    df, event_terms = build_event_dummies(df)

    print("Event-study data preview:")
    print(df[["ADM1_EN", "year", "lnNTL", "Ever_treated", "Treated", "EventTime"]].head(12))

    model = fit_event_study(df, event_terms)
    coefficients = extract_event_coefficients(model)

    print("\nEvent-study coefficients:")
    print(coefficients.to_string(index=False))

    save_outputs(model, coefficients)
    plot_event_study(coefficients, show_plot=show_plot)


if __name__ == "__main__":
    main()
