"""
Robustness and identification checks for regional panel results.

Adds:
- Region-specific linear trends in FE models.
- Driscoll–Kraay covariance.
- Wild cluster bootstrap p-values (clustered by region).
- Placebo/permutation tests (shuffle treated regions; shift treatment timing).
- Sensitivity to alternative leader coding (SNNP/Sidama/SW variants).
- Inclusion vs. exclusion samples side-by-side.
- Event-study lead/lag export.
- Conflict robustness with and without high-conflict years.

Outputs go to outputs/results/regional/robustness/.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.paths import TABULAR_DIR, REGIONAL_RESULTS_DIR  # noqa: E402

PANEL_PATH = TABULAR_DIR / "regional_ntl_panel_data.csv"
CONFLICT_PATH = TABULAR_DIR / "regional_ntl_conflict_panel.csv"
SPECIAL = ["Addis Ababa", "Dire Dawa", "Harari"]

ROBUST_DIR = REGIONAL_RESULTS_DIR / "robustness"
ROBUST_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Data utilities
# ------------------------------
def load_panel(path=PANEL_PATH):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^\s*$")]
    df = df.loc[:, ~df.columns.duplicated()]
    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])
    num_cols = ["year", "NTL", "lnNTL", "Leader_region"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lnNTL", "Leader_region", "ADM1_EN", "year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def load_conflict_panel(path=CONFLICT_PATH):
    df = pd.read_csv(path)
    if "Year" in df.columns and "year" in df.columns:
        df = df.drop(columns=["Year"])
    for c in ["year", "lnNTL", "Leader_region", "events", "fatalities"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lnNTL", "Leader_region", "ADM1_EN", "year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def set_panel_index(df):
    return df.set_index(["ADM1_EN", "year"])


# ------------------------------
# Model helpers
# ------------------------------
def run_fe(df, add_trends=False, cov_type="clustered", cov_config=None, time_effects=True):
    """
    Entity FE + optional year FE + optional entity-specific linear trends (demeaned within entity).
    If add_trends=True, time_effects can be set to False to avoid collinearity.
    """
    cov_config = cov_config or {}
    df_use = df.copy()
    exog = df_use[["Leader_region"]].copy()

    if "events" in df_use.columns and "fatalities" in df_use.columns:
        exog["events"] = df_use["events"]
        exog["fatalities"] = df_use["fatalities"]

    if add_trends:
        trend = df_use.index.get_level_values(1)
        trend_demeaned = trend - pd.Series(trend, index=df_use.index).groupby(level=0).transform("mean")
        exog["trend"] = trend_demeaned

    exog = exog.apply(pd.to_numeric, errors="coerce")
    exog.columns = exog.columns.astype(str)
    mod = PanelOLS(df_use["lnNTL"], exog, entity_effects=True, time_effects=time_effects)
    res = mod.fit(cov_type=cov_type, **cov_config)
    return res


def wild_cluster_bootstrap(df, add_trends=False, reps=20, seed=123):
    """
    Rademacher wild cluster bootstrap on entity clusters for the Leader_region coefficient.
    Returns bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    base_res = run_fe(
        df,
        add_trends=add_trends,
        cov_type="clustered",
        time_effects=not add_trends,
    )
    beta_hat = base_res.params["Leader_region"]

    clusters = df.index.get_level_values(0).unique()
    fitted = base_res.predict().copy()
    resid = base_res.resids.copy()

    boot_stats = []
    for _ in range(reps):
        signs = pd.Series(rng.choice([-1, 1], size=len(clusters)), index=clusters)
        shock = resid.copy().reset_index()
        shock = shock.rename(columns={0: "residual"})
        shock["sign"] = shock["ADM1_EN"].map(signs)
        shock["wild_resid"] = shock["residual"] * shock["sign"]
        shock = shock.set_index(["ADM1_EN", "year"])["wild_resid"]
        shock = shock.reindex(df.index)
        y_star = pd.Series(
            np.asarray(fitted).squeeze() + shock.values, index=df.index
        )
        df_star = df.copy()
        df_star["lnNTL"] = y_star
        res_star = run_fe(
            df_star,
            add_trends=add_trends,
            cov_type="clustered",
            time_effects=not add_trends,
        )
        boot_stats.append(res_star.params["Leader_region"])

    boot_stats = np.array(boot_stats)
    p_val = (np.sum(np.abs(boot_stats) >= np.abs(beta_hat)) + 1) / (reps + 1)
    return beta_hat, p_val


# ------------------------------
# Placebo / permutation
# ------------------------------
def placebo_shuffle_regions(df, reps=20, seed=123):
    rng = np.random.default_rng(seed)
    entities = df.index.get_level_values(0).unique()
    years = df.index.get_level_values(1).unique()
    # panel of treatment by entity-year
    tr_panel = df["Leader_region"].unstack(level=0).reindex(index=years).sort_index()
    results = []
    for _ in range(reps):
        perm = rng.permutation(tr_panel.columns)
        shuffled_panel = tr_panel[perm]
        # stack back to long form matching df index order
        shuffled_long = shuffled_panel.stack().rename("Leader_region").reset_index()
        shuffled_long.columns = ["year", "ADM1_EN", "Leader_region"]
        shuffled_long = shuffled_long.set_index(["ADM1_EN", "year"])
        shuffled_df = df.copy()
        shuffled_df["Leader_region"] = shuffled_long["Leader_region"]
        res = run_fe(shuffled_df, cov_type="clustered")
        results.append(res.params["Leader_region"])
    return np.array(results)


def placebo_shift_timing(df, window=3, reps=20, seed=123):
    rng = np.random.default_rng(seed)
    results = []
    df_reset = df.reset_index()
    treated_regions = (
        df_reset.loc[df_reset["Leader_region"] == 1, "ADM1_EN"].unique().tolist()
    )
    for _ in range(reps):
        shifted = df_reset.copy()
        for region in treated_regions:
            region_rows = shifted["ADM1_EN"] == region
            years = shifted.loc[region_rows, "year"]
            shift = rng.integers(-window, window + 1)
            shifted.loc[region_rows, "Leader_region"] = 0
            shifted.loc[region_rows & shifted["year"].isin(years + shift), "Leader_region"] = 1
        shifted = shifted.set_index(["ADM1_EN", "year"])
        res = run_fe(shifted, cov_type="clustered")
        results.append(res.params["Leader_region"])
    return np.array(results)


# ------------------------------
# Alternative leader coding
# ------------------------------
def recode_leader(row, scheme):
    y = row["year"]
    region = row["ADM1_EN"]

    # Baseline: Tigray 1992-2012; SNNP/Sidama/SW 2012-2018; Oromia 2018-2024
    if scheme == "baseline":
        if 1992 <= y <= 2012 and region == "Tigray":
            return 1
        if 2012 <= y <= 2018 and region in ["SNNP", "Sidama", "South West Ethiopia"]:
            return 1
        if 2018 <= y <= 2024 and region == "Oromia":
            return 1
        return 0

    # SNNP only (exclude Sidama, SW successors from treatment years)
    if scheme == "snnponly":
        if 1992 <= y <= 2012 and region == "Tigray":
            return 1
        if 2012 <= y <= 2018 and region == "SNNP":
            return 1
        if 2018 <= y <= 2024 and region == "Oromia":
            return 1
        return 0

    # Successor-specific: Sidama treated from 2019+, SW Ethiopia from 2021+
    if scheme == "successors_split":
        if 1992 <= y <= 2012 and region == "Tigray":
            return 1
        if 2012 <= y <= 2018 and region == "SNNP":
            return 1
        if 2019 <= y <= 2024 and region == "Sidama":
            return 1
        if 2021 <= y <= 2024 and region == "South West Ethiopia":
            return 1
        if 2018 <= y <= 2024 and region == "Oromia":
            return 1
        return 0

    raise ValueError(f"Unknown scheme: {scheme}")


def run_leader_coding_sensitivity(df):
    schemes = ["baseline", "snnponly", "successors_split"]
    out_rows = []
    for scheme in schemes:
        df_temp = df.reset_index()
        df_temp["Leader_region_alt"] = df_temp.apply(
            lambda r: recode_leader(r, scheme), axis=1
        )
        df_full = df_temp.rename(columns={"Leader_region_alt": "Leader_region"})
        model_full = sm.OLS.from_formula(
            "lnNTL ~ Leader_region + C(ADM1_EN) + C(year)", data=df_full
        ).fit(cov_type="cluster", cov_kwds={"groups": df_full["ADM1_EN"]})
        df_excl = df_full.loc[~df_full["ADM1_EN"].isin(SPECIAL)].copy()
        model_excl = sm.OLS.from_formula(
            "lnNTL ~ Leader_region + C(ADM1_EN) + C(year)", data=df_excl
        ).fit(cov_type="cluster", cov_kwds={"groups": df_excl["ADM1_EN"]})
        def pull_coef(model):
            name = [k for k in model.params.index if k.startswith("Leader_region")][0]
            return model.params[name], model.bse[name]

        coef_full, se_full = pull_coef(model_full)
        coef_excl, se_excl = pull_coef(model_excl)

        out_rows.append(
            {"scheme": scheme, "sample": "full", "coef": coef_full, "se": se_full}
        )
        out_rows.append(
            {
                "scheme": scheme,
                "sample": "excl_special",
                "coef": coef_excl,
                "se": se_excl,
            }
        )
    return pd.DataFrame(out_rows)


# ------------------------------
# Conflict robustness
# ------------------------------
def drop_high_conflict_years(df, quantile=0.95):
    yearly = df.reset_index().groupby("year")[["events", "fatalities"]].sum()
    cutoff = yearly["events"].quantile(quantile)
    keep_years = yearly.loc[yearly["events"] <= cutoff].index.tolist()
    return df.loc[df.index.get_level_values(1).isin(keep_years)]


# ------------------------------
# Event-study export
# ------------------------------
def export_event_study(df, output_path):
    # Build leads/lags as in main event-study script (window -5 to +5, omit -1)
    df_es = df.reset_index()
    treated = df_es[df_es["Leader_region"] == 1].groupby("ADM1_EN")["year"].min()
    df_es["t0"] = df_es["ADM1_EN"].map(treated)
    df_es["event_time"] = df_es["year"] - df_es["t0"]
    for k in range(-5, 6):
        if k == -1:
            continue
        col = f"event{k}" if k >= 0 else f"event_m{abs(k)}"
        df_es[col] = (df_es["event_time"] == k).astype(int) * (
            df_es["Leader_region"] == 1
        )

    event_terms = " + ".join(
        [f"event_m{abs(k)}" if k < 0 else f"event{k}" for k in range(-5, 6) if k != -1]
    )
    formula = f"lnNTL ~ {event_terms} + C(ADM1_EN) + C(year)"
    model = sm.OLS.from_formula(formula, data=df_es).fit(
        cov_type="cluster", cov_kwds={"groups": df_es["ADM1_EN"]}
    )

    rows = []
    for k in range(-5, 6):
        if k == -1:
            continue
        name = f"event_m{abs(k)}" if k < 0 else f"event{k}"
        rows.append(
            {
                "event_time": k,
                "coef": model.params.get(name, np.nan),
                "se": model.bse.get(name, np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return model


# ------------------------------
# Main runner
# ------------------------------
def main():
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)

    # Base panel
    df = set_panel_index(load_panel())
    df_excl = df.loc[~df.index.get_level_values(0).isin(SPECIAL)]

    # 1) Region trends FE
    res_trend_full = run_fe(df, add_trends=True, cov_type="clustered", time_effects=False)
    res_trend_excl = run_fe(df_excl, add_trends=True, cov_type="clustered", time_effects=False)
    with open(ROBUST_DIR / "fe_with_trends.txt", "w", encoding="utf-8") as f:
        f.write("Full sample: entity+year FE with region trends (clustered)\n")
        f.write(str(res_trend_full.summary))
        f.write("\n\nExcluding Addis/Dire/Harari\n")
        f.write(str(res_trend_excl.summary))

    # 2) Driscoll–Kraay
    dk_kw = {"kernel": "bartlett", "bandwidth": 3}
    res_dk_full = run_fe(df, cov_type="driscoll-kraay", cov_config=dk_kw)
    res_dk_excl = run_fe(df_excl, cov_type="driscoll-kraay", cov_config=dk_kw)
    with open(ROBUST_DIR / "fe_driscoll_kraay.txt", "w", encoding="utf-8") as f:
        f.write("Full sample: entity+year FE (Driscoll–Kraay)\n")
        f.write(str(res_dk_full.summary))
        f.write("\n\nExcluding Addis/Dire/Harari\n")
        f.write(str(res_dk_excl.summary))

    # 3) Wild cluster bootstrap
    beta_full, p_full = wild_cluster_bootstrap(df, add_trends=False, reps=20)
    beta_trend_full, p_trend_full = wild_cluster_bootstrap(
        df, add_trends=True, reps=20
    )
    with open(ROBUST_DIR / "fe_wild_cluster_bootstrap.txt", "w", encoding="utf-8") as f:
        f.write("Wild cluster bootstrap (20 reps, Rademacher, cluster=entity)\n")
        f.write(f"FE+year FE beta={beta_full:.3f}, p={p_full:.3f}\n")
        f.write(
            f"FE+year FE + trends beta={beta_trend_full:.3f}, p={p_trend_full:.3f}\n"
        )

    # 4) Placebo tests
    placebo_reg = placebo_shuffle_regions(df, reps=20)
    placebo_time = placebo_shift_timing(df, window=3, reps=20)
    pd.Series(placebo_reg).to_csv(ROBUST_DIR / "placebo_shuffle_regions.csv", index=False)
    pd.Series(placebo_time).to_csv(ROBUST_DIR / "placebo_shift_timing.csv", index=False)

    # 5) Leader coding sensitivity
    sens = run_leader_coding_sensitivity(df)
    sens.to_csv(ROBUST_DIR / "leader_coding_sensitivity.csv", index=False)

    # 6) Conflict robustness (with and without high-conflict years)
    df_conf = set_panel_index(load_conflict_panel())
    res_conf_full = run_fe(df_conf, cov_type="clustered")
    res_conf_excl = run_fe(
        df_conf.loc[~df_conf.index.get_level_values(0).isin(SPECIAL)],
        cov_type="clustered",
    )
    with open(ROBUST_DIR / "fe_conflict_joint.txt", "w", encoding="utf-8") as f:
        f.write("Full sample with conflict controls (clustered)\n")
        f.write(str(res_conf_full.summary))
        f.write("\n\nExcluding Addis/Dire/Harari with conflict controls\n")
        f.write(str(res_conf_excl.summary))

    df_conf_trim = drop_high_conflict_years(df_conf, quantile=0.95)
    res_conf_trim = run_fe(df_conf_trim, cov_type="clustered")
    with open(
        ROBUST_DIR / "fe_conflict_drop_high_years.txt", "w", encoding="utf-8"
    ) as f:
        f.write("Full sample with conflict controls, dropping high-conflict years\n")
        f.write(str(res_conf_trim.summary))

    # 7) Event-study export
    export_event_study(df, ROBUST_DIR / "event_study_coefficients.csv")


if __name__ == "__main__":
    main()
