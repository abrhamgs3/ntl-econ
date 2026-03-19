import pandas as pd

def extract_gdp_timeseries(gdp_csv_path, output_path):
    # Read the GDP CSV file
    df = pd.read_csv(gdp_csv_path)
    # Only keep Year and GDP columns (GDPPC_con)
    gdp_df = df[['Year', 'GDPPC_con']].copy()
    # Remove rows with missing GDP
    gdp_df = gdp_df.dropna(subset=['GDPPC_con'])
    # Save to Excel
    gdp_df.to_excel(output_path, index=False)
    print(f'GDP time series saved to: {output_path}')

if __name__ == '__main__':
    extract_gdp_timeseries(
        gdp_csv_path=r'../Nighttime Light Paper/01 Data and Syntax/Eth_1992_2023_final GDP.csv',
        output_path=r'data/tabular/ethiopia_gdp_timeseries.xlsx'
    )
