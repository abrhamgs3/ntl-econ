import pandas as pd
from pathlib import Path

INPUT_FILE = "data/raw/land_use_data.csv"  # Replace with actual file path
OUTPUT_FILE = Path("data/processed/land_use_processed.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def process_land_use_data():
    """Process land use data."""
    land_use_data = pd.read_csv(INPUT_FILE)
    # Example processing: aggregate by year and region
    processed_data = land_use_data.groupby(["Year", "Region"]).sum().reset_index()
    processed_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Processed land use data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_land_use_data()