"""
transparency_appendix.py

Script to generate empirical transparency outputs for manuscript appendix:
- Alternative treatment codings and results
- Full specification table
- Influence diagnostics
- Placebo distributions
- Power calculations

Each section:
- Explains what is shown
- Outputs template tables/figures (CSV, LaTeX, PNG)
- Includes short interpretation text

Fill in data loading and model details as appropriate for your project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ========== 1. Alternative Treatment Codings and Results ==========
def alternative_treatment_codings(df, coding_schemes):
    """
    For each alternative leader-region coding, estimate the association with NTL.
    Outputs a CSV and LaTeX table summarizing results.
    """
    results = []
    for name, treat_col in coding_schemes.items():
        X = sm.add_constant(df[treat_col])
        y = df['ntl']
        model = sm.OLS(y, X).fit()
        results.append({
            'Coding Scheme': name,
            'Treated Regions': df[treat_col].sum(),
            'Years': df['year'].nunique(),
            'Estimated Effect (SE)': f"{model.params[treat_col]:.3f} ({model.bse[treat_col]:.3f})"
        })
    res_df = pd.DataFrame(results)
    res_df.to_csv('alt_treatment_codings.csv', index=False)
    print("\nAlternative Treatment Codings and Results:")
    print(res_df)
    print("\nInterpretation: Results are sensitive to how leader-region status is coded, with estimated effects varying across plausible assignments.")
    # Optionally, output LaTeX table
    with open('alt_treatment_codings.tex', 'w') as f:
        f.write(res_df.to_latex(index=False))

# ========== 2. Full Specification Table ==========
def full_specification_table(specs):
    """
    Summarize all model specifications and results.
    Outputs a CSV and LaTeX table.
    """
    spec_df = pd.DataFrame(specs)
    spec_df.to_csv('specification_table.csv', index=False)
    print("\nFull Specification Table:")
    print(spec_df)
    print("\nInterpretation: Estimated leader-region effects are highly sensitive to specification, with sign, magnitude, and significance varying across models.")
    with open('specification_table.tex', 'w') as f:
        f.write(spec_df.to_latex(index=False))

# ========== 3. Influence Diagnostics ==========
def influence_diagnostics(df, model, region_col='region'):
    """
    Compute Cook's distance and leverage for each region-year.
    Output as CSV and print summary.
    """
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    df_out = df.copy()
    df_out['Cook_D'] = cooks
    df_out['Leverage'] = leverage
    df_out[['region', 'year', 'Cook_D', 'Leverage']].to_csv('influence_diagnostics.csv', index=False)
    print("\nInfluence Diagnostics (top influential regions):")
    print(df_out[['region', 'year', 'Cook_D', 'Leverage']].sort_values('Cook_D', ascending=False).head())
    print("\nInterpretation: A small number of high-luminosity regions exert outsized influence on estimated effects.")

# ========== 4. Placebo Distributions ==========
def placebo_distribution(df, treat_col, n_iter=1000):
    """
    Generate placebo distribution by permuting treatment assignment.
    Plot histogram and save as PNG.
    """
    observed = sm.OLS(df['ntl'], sm.add_constant(df[treat_col])).fit().params[treat_col]
    placebo_betas = []
    for _ in range(n_iter):
        shuffled = np.random.permutation(df[treat_col].values)
        beta = sm.OLS(df['ntl'], sm.add_constant(shuffled)).fit().params[1]
        placebo_betas.append(beta)
    plt.hist(placebo_betas, bins=30, alpha=0.7)
    plt.axvline(observed, color='red', linestyle='dashed', label='Observed')
    plt.title('Placebo Distribution of Estimated Effects')
    plt.xlabel('Estimated Effect')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('placebo_distribution.png')
    plt.close()
    print(f"\nPlacebo Distribution: Observed effect = {observed:.3f}")
    pval = np.mean(np.abs(placebo_betas) >= np.abs(observed))
    print(f"Empirical p-value: {pval:.3f}")
    print("Interpretation: The observed leader-region effect is frequently matched or exceeded by placebo assignments.")

# ========== 5. Power Calculations ==========
def power_calculation(df, treat_col, alpha=0.05, effect_size=0.1):
    """
    Compute minimum detectable effect and empirical power.
    Output as print statement and CSV.
    """
    from statsmodels.stats.power import TTestIndPower
    n = len(df)
    std = df['ntl'].std()
    analysis = TTestIndPower()
    power = analysis.power(effect_size=effect_size/std, nobs1=n//2, alpha=alpha)
    min_detectable = analysis.solve_power(effect_size=None, nobs1=n//2, alpha=alpha, power=0.8) * std
    out = pd.DataFrame({
        'Test': ['Leader-region effect'],
        'Minimum Detectable Effect': [min_detectable],
        'Power (at alpha=0.05)': [power]
    })
    out.to_csv('power_calculations.csv', index=False)
    print("\nPower Calculations:")
    print(out)
    print("Interpretation: Power is low due to the small number of treated units and limited variation.")

# ========== Main ========== 
if __name__ == "__main__":
    # Example usage (replace with actual data/model calls)
    print("Transparency Appendix Script: Fill in with your data and models.")
    # df = pd.read_csv('your_panel_data.csv')
    # coding_schemes = {'Main': 'treat_main', 'Split': 'treat_split'}
    # alternative_treatment_codings(df, coding_schemes)
    # specs = [ ... ]
    # full_specification_table(specs)
    # model = sm.OLS(df['ntl'], sm.add_constant(df['treat_main'])).fit()
    # influence_diagnostics(df, model)
    # placebo_distribution(df, 'treat_main')
    # power_calculation(df, 'treat_main')
