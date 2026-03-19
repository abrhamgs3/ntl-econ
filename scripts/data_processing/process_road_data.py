import geopandas as gpd
from pathlib import Path

INPUT_FILE = "data/raw/road_network.shp"  # Replace with actual file path
OUTPUT_FILE = Path("data/processed/roads_processed.shp")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def process_road_data():
    """Process road infrastructure data."""
    road_data = gpd.read_file(INPUT_FILE)
    # Example processing: filter by road type
    processed_data = road_data[road_data["road_type"].isin(["Highway", "Main Road"])]
    processed_data.to_file(OUTPUT_FILE)
    print(f"Processed road data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_road_data()