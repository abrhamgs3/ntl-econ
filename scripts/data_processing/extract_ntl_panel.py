import glob
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from scripts.common.paths import CLIPPED_RASTER_DIR, SHAPEFILES_DIR, TABULAR_DIR

SHAPEFILE_PATH = SHAPEFILES_DIR / "eth_admbnda_adm1_csa_bofedb_2021.shp"
REGIONAL_OUTPUT_CSV = TABULAR_DIR / "regional_ntl_panel_data.csv"
REGIONAL_OUTPUT_XLSX = TABULAR_DIR / "regional_ntl_panel_data.xlsx"
NATIONAL_OUTPUT = TABULAR_DIR / "ethiopia_ntl_data.xlsx"
LOG_OFFSET = 0.01

LEADER_PERIODS = {
    "Tigray": (1992, 2012),
    "SNNP": (2012, 2018),
    "Sidama": (2012, 2018),
    "South West Ethiopia": (2012, 2018),
    "Oromia": (2018, 2024),
}


def extract_year_from_filename(filename):
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group(0)) if match else None


def assign_leader_region(region_name, year):
    period = LEADER_PERIODS.get(region_name)
    if period is None:
        return 0

    start_year, end_year = period
    return int(start_year <= year <= end_year)


def load_regions(shapefile_path=SHAPEFILE_PATH):
    regions = gpd.read_file(shapefile_path)
    if "ADM1_EN" not in regions.columns:
        raise KeyError("Expected 'ADM1_EN' in shapefile attributes.")
    return regions


def get_raster_files(clipped_raster_dir=CLIPPED_RASTER_DIR):
    raster_files = sorted(glob.glob(str(clipped_raster_dir / "Harmonized_DN_NTL_*.tif")))
    if not raster_files:
        raise FileNotFoundError(f"No NTL raster files found in {clipped_raster_dir}")
    return raster_files


def build_regional_panel(regions, raster_files):
    rows = []

    for raster_path in raster_files:
        year = extract_year_from_filename(raster_path)
        if year is None:
            print(f"Skipping file with no year in name: {raster_path}")
            continue

        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata if src.nodata is not None else -999
            stats = zonal_stats(regions, raster_path, stats=["mean", "sum"], nodata=nodata_value)

        for region, stat in zip(regions.itertuples(), stats):
            mean_ntl = 0.0 if stat["mean"] is None else float(stat["mean"])
            total_ntl = 0.0 if stat["sum"] is None else float(stat["sum"])
            rows.append(
                {
                    "ADM1_EN": region.ADM1_EN,
                    "Year": year,
                    "Mean_NTL": mean_ntl,
                    "NTL": total_ntl,
                    "Leader_region": assign_leader_region(region.ADM1_EN, year),
                }
            )

    regional_df = pd.DataFrame(rows)
    regional_df["year"] = regional_df["Year"]
    regional_df["lnNTL"] = np.log(regional_df["NTL"] + LOG_OFFSET)
    regional_df = regional_df.sort_values(["ADM1_EN", "year"]).reset_index(drop=True)
    return regional_df


def build_national_series(raster_files):
    rows = []

    for raster_path in raster_files:
        year = extract_year_from_filename(raster_path)
        if year is None:
            continue

        with rasterio.open(raster_path) as src:
            mean_ntl = float(src.read(1).mean())
        rows.append({"Year": year, "Mean_NTL": mean_ntl})

    national_df = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
    return national_df


def merge_existing_gdp(national_df, existing_path=NATIONAL_OUTPUT):
    """
    Preserve an existing GDP series if the workbook already contains it.
    """
    if not existing_path.exists():
        return national_df

    try:
        existing_df = pd.read_excel(existing_path)
    except Exception:
        return national_df

    if {"Year", "GDPPC_con"}.issubset(existing_df.columns):
        gdp_df = existing_df[["Year", "GDPPC_con"]].dropna().drop_duplicates(subset=["Year"])
        national_df = national_df.merge(gdp_df, on="Year", how="left")

    return national_df


def save_outputs(regional_df, national_df):
    TABULAR_DIR.mkdir(parents=True, exist_ok=True)
    regional_df.to_csv(REGIONAL_OUTPUT_CSV, index=False)
    regional_df.to_excel(REGIONAL_OUTPUT_XLSX, index=False)
    national_df.to_excel(NATIONAL_OUTPUT, index=False)

    print(f"Saved regional panel to: {REGIONAL_OUTPUT_CSV}")
    print(f"Saved regional panel workbook to: {REGIONAL_OUTPUT_XLSX}")
    print(f"Saved national series to: {NATIONAL_OUTPUT}")


def main():
    print("Loading shapefile...")
    regions = load_regions()
    print(f"Regions loaded: {len(regions)}")

    raster_files = get_raster_files()
    print(f"Found {len(raster_files)} clipped NTL rasters.")

    print("Building regional panel...")
    regional_df = build_regional_panel(regions, raster_files)
    print(regional_df.head())

    print("Building national series...")
    national_df = build_national_series(raster_files)
    national_df = merge_existing_gdp(national_df)
    print(national_df.head())

    save_outputs(regional_df, national_df)


if __name__ == "__main__":
    main()
