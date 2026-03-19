import requests
import pandas as pd
from pathlib import Path

ACLED_URL = "https://api.acleddata.com/acled/read"  # Replace with actual API endpoint
API_KEY = "your_api_key"  # Replace with your API key
OUTPUT_DIR = Path("data/processed/conflict")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_acled_data():
    """Fetch conflict data from ACLED API."""
    params = {
        "key": API_KEY,
        "email": "your_email@example.com",  # Replace with your email
        "country": "Ethiopia",
        "event_date": "1992-01-01|2025-12-31",
        "format": "json",
    }
    response = requests.get(ACLED_URL, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["data"])
    output_file = OUTPUT_DIR / "ethiopia_conflict_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Conflict data saved to {output_file}")

if __name__ == "__main__":
    fetch_acled_data()