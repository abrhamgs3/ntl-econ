import re
from pathlib import Path

from statsmodels.stats.stattools import durbin_watson

from scripts.analysis.analyze_ntl_gdp import load_data, prepare_regression_sample, run_regression
from scripts.analysis.analyze_regional_panel import compute_gini_by_year, load_and_clean_data
from scripts.analysis.analyze_regional_panel_exclude_special import (
    load_and_clean_data as load_excluding_special,
)
from scripts.analysis.analyze_regional_panel_exclude_special import (
    prepare_panel as prepare_panel_excluding_special,
)
from scripts.analysis.analyze_regional_panel_exclude_special import (
    run_panel_regressions as run_excluding_special_regressions,
)
from scripts.common.paths import MANUSCRIPT_DIR, TABULAR_DIR


TABLES_DIR = MANUSCRIPT_DIR / "tables"
KEY_RESULTS_TABLE_PATH = TABLES_DIR / "key_results.tex"
ARCHIVED_EXCL_ADDIS_PATH = TABULAR_DIR / "regional_ntl_birthregion_panel_summary_excl_addis.txt"
ARCHIVED_INCL_ADDIS_PATH = TABULAR_DIR / "regional_ntl_birthregion_panel_summary_incl_addis.txt"


def parse_archived_leader_result(path):
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Leader_Birth_Region\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)"
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse archived leader result from {path}")

    coefficient = float(match.group(1))
    p_value = float(match.group(4))
    return coefficient, p_value


def build_key_results_table():
    national_df = load_data()
    national_reg_df = prepare_regression_sample(national_df)
    national_model, national_corr, national_elasticity = run_regression(national_reg_df)

    regional_df = load_and_clean_data()
    gini_by_year = compute_gini_by_year(regional_df).set_index("year")

    archived_excl_coef, archived_excl_p = parse_archived_leader_result(ARCHIVED_EXCL_ADDIS_PATH)
    archived_incl_coef, archived_incl_p = parse_archived_leader_result(ARCHIVED_INCL_ADDIS_PATH)

    excl_special_df = load_excluding_special()
    excl_special_panel = prepare_panel_excluding_special(excl_special_df)
    excl_special_results = run_excluding_special_regressions(excl_special_panel)
    within_result = excl_special_results["(3) Within"]
    dw_stat = durbin_watson(national_model.resid)

    table = rf"""\begin{{table}}[ht]
\centering
\caption{{Key quantitative findings from the cleaned analysis pipeline}}
\begin{{tabular}}{{ll}}
\toprule
Statistic & Value \\
\midrule
Correlation between mean NTL and log GDP per capita & {national_corr:.3f} \\
Log-log elasticity of NTL with respect to GDP per capita & {national_elasticity:.3f} \\
National validation regression $R^2$ & {national_model.rsquared:.3f} \\
Durbin-Watson statistic & {dw_stat:.3f} \\
Regional Gini coefficient, 1992 & {gini_by_year.loc[1992, "gini"]:.3f} \\
Regional Gini coefficient, 2018 & {gini_by_year.loc[2018, "gini"]:.3f} \\
Regional Gini coefficient, 2024 & {gini_by_year.loc[2024, "gini"]:.3f} \\
Archived FE coefficient on leader birth-region indicator (excl. Addis Ababa) & {archived_excl_coef:.3f} \\
$p$-value for archived FE coefficient (excl. Addis Ababa) & {archived_excl_p:.3f} \\
Archived FE coefficient on leader birth-region indicator (incl. Addis Ababa) & {archived_incl_coef:.3f} \\
$p$-value for archived FE coefficient (incl. Addis Ababa) & {archived_incl_p:.3f} \\
Within estimate excluding Addis Ababa, Dire Dawa, and Harari & {within_result.params["Leader_region"]:.3f} \\
$p$-value for within estimate excluding special regions & {within_result.pvalues["Leader_region"]:.3f} \\
\bottomrule
\end{{tabular}}
\label{{tab:keyresults}}
\end{{table}}
"""
    return table


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_text = build_key_results_table()
    KEY_RESULTS_TABLE_PATH.write_text(table_text, encoding="utf-8")
    print(f"Saved manuscript table to: {KEY_RESULTS_TABLE_PATH}")


if __name__ == "__main__":
    main()
