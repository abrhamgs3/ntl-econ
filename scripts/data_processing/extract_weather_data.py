import requests
import pandas as pd
from pathlib import Path

WEATHER_API_URL = "https://api.weather.com/v1"  # Replace with actual API endpoint
API_KEY = "your_api_key"  # Replace with your API key
OUTPUT_DIR = Path("data/processed/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_weather_data():
    """Fetch weather data from API."""
    params = {
        "apiKey": API_KEY,
        "country": "Ethiopia",
        "startDate": "1992-01-01",
        "endDate": "2025-12-31",
        "format": "json",
    }
    response = requests.get(WEATHER_API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["observations"])
    output_file = OUTPUT_DIR / "ethiopia_weather_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Weather data saved to {output_file}")

if __name__ == "__main__":
    fetch_weather_data()