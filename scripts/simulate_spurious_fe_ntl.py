import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
n_regions = 13
n_years = 20
regions = [f'region_{i+1}' for i in range(n_regions)]
years = np.arange(2000, 2000 + n_years)

# Simulate region effects (one dominant region)
mu = np.array([100] + [10]*(n_regions-1))

# Simulate data
records = []
for i, region in enumerate(regions):
    for t, year in enumerate(years):
        region_effect = mu[i]
        year_effect = 5 * t  # deterministic trend
        eps = np.random.normal(0, 1)
        log_ntl = np.log(region_effect) + year_effect/100 + eps
        records.append({'region': region, 'year': year, 'log_ntl': log_ntl})
df = pd.DataFrame(records)


# Random leader assignment: assign one leader region per year
leader_for_year = pd.DataFrame({
    'year': years,
    'leader_region': np.random.choice(regions, size=n_years)
})
# Merge leader assignment onto main DataFrame
df = df.merge(leader_for_year, on='year', how='left')
# Indicator for whether region is leader in that year
df['D'] = (df['region'] == df['leader_region']).astype(int)

# FE model (region FE only)
model1 = smf.ols('log_ntl ~ C(region) + D', data=df).fit()
print('Region FE only:')
print(model1.summary())

# FE model (region + year FE)
model2 = smf.ols('log_ntl ~ C(region) + C(year) + D', data=df).fit()
print('\nRegion + Year FE:')
print(model2.summary())

# Weighted FE model (region FE, weights inverse mean luminosity)
region_means = df.groupby('region')['log_ntl'].mean()
weights = df['region'].map(lambda r: 1/region_means[r])
model3 = smf.ols('log_ntl ~ C(region) + D', data=df, weights=weights).fit()
print('\nWeighted Region FE:')
print(model3.summary())

# Plot leader-region coefficient across models
coefs = [model1.params['D'], model2.params.get('D', np.nan), model3.params['D']]
labels = ['Region FE', 'Region+Year FE', 'Weighted FE']
plt.bar(labels, coefs)
plt.ylabel('Leader-region coefficient')
plt.title('Spurious Leader-Region Effects in Simulated NTL Data')
plt.show()
