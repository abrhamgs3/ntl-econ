import os
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path

# Define constants
DMSP_URL = "https://ngdc.noaa.gov/eog/dmsp/downloadV4composites/"  # NOAA NCEI DMSP data source
VIIRS_URL = "https://eogdata.mines.edu/nighttime_light/annual/v20/"  # Earth Observation Group VIIRS data source
YEARS = range(1992, 2026)
COUNTRY_SHAPEFILE = "data/shapefiles/eth_admbnda_adm0_csa_bofedb_itos_2021.shp"
REGIONS_SHAPEFILE = "data/shapefiles/eth_admbnda_adm1_csa_bofedb_2021.shp"
OUTPUT_DIR = Path("data/processed/ntl")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clip_ntl_data(raster_path, shapefile_path, output_path):
    """Clip NTL raster data to the boundaries of a shapefile."""
    with rasterio.open(raster_path) as src:
        shapefile = gpd.read_file(shapefile_path)
        shapes = [feature["geometry"] for feature in shapefile.iterfeatures()]
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def process_ntl_data():
    """Process manually downloaded NTL data for all years."""
    for year in YEARS:
        for dataset in ["VIIRS"]:  # Process only VIIRS data
            print(f"Processing {dataset} data for {year}...")
            raster_path = OUTPUT_DIR / f"{dataset}_{year}.tif"

            if not raster_path.exists():
                print(f"File not found: {raster_path}. Please ensure the file is manually downloaded.")
                continue

            # Clip for country
            country_output = OUTPUT_DIR / f"{dataset}_{year}_country.tif"
            clip_ntl_data(raster_path, COUNTRY_SHAPEFILE, country_output)

            # Clip for regions
            regions_output = OUTPUT_DIR / f"{dataset}_{year}_regions.tif"
            clip_ntl_data(raster_path, REGIONS_SHAPEFILE, regions_output)

if __name__ == "__main__":
    process_ntl_data()
    print("NTL data processing complete.")