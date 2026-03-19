"""
Simulate panel data to illustrate identification failure in FE models with varying N, treatment share, and outcome concentration.
Generates plots of bias and variance for FE estimates with and without time fixed effects.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

def simulate_panel(N=13, T=20, treat_share=0.2, heavy_tail=False, seed=None):
    np.random.seed(seed)
    # Assign treatment to a subset of units and periods
    treat_units = np.random.choice(N, int(N * treat_share), replace=False)
    treat_matrix = np.zeros((N, T))
    for i in treat_units:
        treat_matrix[i, np.random.randint(0, T//2):] = 1  # treatment starts mid-panel
    # Generate time shocks
    time_shock = np.random.normal(0, 1, T)
    # Generate unit effects
    if heavy_tail:
        unit_effect = np.random.standard_t(df=2, size=N) * 2  # heavy-tailed
    else:
        unit_effect = np.random.normal(0, 1, N)
    # Generate outcome
    beta = 1.0  # true effect
    Y = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            Y[i, t] = (
                unit_effect[i] + time_shock[t] + beta * treat_matrix[i, t] + np.random.normal(0, 1)
            )
    return pd.DataFrame({
        'unit': np.repeat(np.arange(N), T),
        'time': np.tile(np.arange(T), N),
        'D': treat_matrix.flatten(),
        'Y': Y.flatten()
    })

def fe_estimate(df, time_fe=False):
    # Two-way FE: Y ~ D + unit FE (+ time FE)
    dummies = pd.get_dummies(df['unit'], prefix='unit', drop_first=True)
    X = pd.concat([df[['D']], dummies], axis=1)
    if time_fe:
        t_dummies = pd.get_dummies(df['time'], prefix='time', drop_first=True)
        X = pd.concat([X, t_dummies], axis=1)
    X = sm.add_constant(X)
    X = X.astype(float)
    y = df['Y'].astype(float)
    model = sm.OLS(y, X).fit()
    return model.params['D'], model.bse['D']

def run_simulation_grid(N_grid, treat_grid, heavy_tail_opts, nrep=100, T=20, seed=42, out_dir="outputs/figures/simulation"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for N in N_grid:
        for treat_share in treat_grid:
            for heavy_tail in heavy_tail_opts:
                for rep in range(nrep):
                    df = simulate_panel(N=N, T=T, treat_share=treat_share, heavy_tail=heavy_tail, seed=seed+rep)
                    for time_fe in [False, True]:
                        est, se = fe_estimate(df, time_fe=time_fe)
                        results.append({
                            'N': N, 'treat_share': treat_share, 'heavy_tail': heavy_tail,
                            'time_fe': time_fe, 'est': est, 'se': se
                        })
    res_df = pd.DataFrame(results)
    # Plot bias and variance
    for heavy_tail in heavy_tail_opts:
        for time_fe in [False, True]:
            for treat_share in treat_grid:
                subset = res_df[(res_df['heavy_tail']==heavy_tail) & (res_df['time_fe']==time_fe) & (res_df['treat_share']==treat_share)]
                means = subset.groupby('N')['est'].mean()
                stds = subset.groupby('N')['est'].std()
                plt.figure(figsize=(7,4))
                plt.plot(means.index, means.values, label='Bias (mean)', marker='o')
                plt.plot(stds.index, stds.values, label='Std (variance)', marker='s')
                plt.axhline(1.0, color='gray', linestyle='--', label='True Effect')
                plt.xlabel('N (Number of Units)')
                plt.ylabel('Estimate')
                plt.title(f"FE {'+Time FE' if time_fe else 'No Time FE'} | Treat share={treat_share} | Heavy tail={heavy_tail}")
                plt.legend()
                fname = f"sim_biasvar_N{treat_share}_ht{heavy_tail}_fe{time_fe}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname), dpi=300)
                plt.close()
    return res_df

if __name__ == "__main__":
    N_grid = [8, 13, 20, 40]
    treat_grid = [0.15, 0.3, 0.5]
    heavy_tail_opts = [False, True]
    run_simulation_grid(N_grid, treat_grid, heavy_tail_opts)
