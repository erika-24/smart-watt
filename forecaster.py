import os
import pandas as pd
import pickle as cPickle
import requests
from itertools import zip_longest

class SolcastWeatherForecast:
    def __init__(self, api_key, rooftop_ids, forecast_hours=48, cache_path="weather_forecast_cache.pkl"):
        """
        Initialize the Solcast API Forecasting class.

        :param api_key: Solcast API key.
        :param rooftop_ids: A list of rooftop site IDs registered with Solcast.
        :param forecast_hours: Number of forecast hours to fetch (default: 48).
        :param cache_path: Path for storing cached forecast data.
        """
        self.api_key = api_key
        self.rooftop_ids = rooftop_ids if isinstance(rooftop_ids, list) else [rooftop_ids]
        self.forecast_hours = forecast_hours
        self.cache_path = os.path.abspath(cache_path)

    def fetch_forecast(self):
        """
        Fetch weather forecast data from Solcast API.

        :return: DataFrame containing the forecasted solar power data.
        """
        headers = {
            "User-Agent": "Solcast-Forecaster",
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        total_data_list = []
        timestamps = None

        for rooftop_id in self.rooftop_ids:
            url = f"https://api.solcast.com.au/rooftop_sites/{rooftop_id}/forecasts?hours={self.forecast_hours}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
            elif response.status_code in [402, 429]:
                raise Exception("Solcast API limit exceeded. Check your subscription.")
            elif response.status_code >= 400:
                raise Exception("Solcast API request failed. Check API key and rooftop ID.")

            forecast_values = [entry["pv_estimate"] * 1000 for entry in data["forecasts"]]
            timestamps = [entry["period_end"] for entry in data["forecasts"]]

            if len(forecast_values) < self.forecast_hours:
                raise Exception("Not enough forecast data received from Solcast.")

            if not total_data_list:
                total_data_list = forecast_values
            else:
                total_data_list = [
                    total + current for total, current in zip_longest(total_data_list, forecast_values, fillvalue=0)
                ]

        # Convert timestamps to DataFrame
        forecast_df = pd.DataFrame({"timestamp": timestamps, "solar_power_watts": total_data_list})
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        forecast_df.set_index("timestamp", inplace=True)

        # Save to cache
        self.save_cache(forecast_df)

        return forecast_df

    def save_cache(self, data):
        """
        Save forecast data to a local cache file.

        :param data: DataFrame containing forecast data.
        """
        with open(self.cache_path, "wb") as file:
            cPickle.dump(data, file)

    def load_cache(self):
        """
        Load cached forecast data if available.

        :return: Cached DataFrame or None if cache is not found.
        """
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "rb") as file:
                data = cPickle.load(file)
                return data
        return None

    def get_forecast(self, use_cache=True):
        """
        Get the forecast data, using cache if available.

        :param use_cache: Whether to use cached data if available.
        :return: DataFrame with forecasted solar power.
        """
        if use_cache:
            cached_data = self.load_cache()
            if cached_data is not None:
                print("Loaded forecast data from cache.")
                return cached_data

        print("Fetching new forecast data from Solcast...")
        return self.fetch_forecast()



if __name__ == "__main__":
    API_KEY = "zozFGx0b2laj8j4Bqq1SVtUV2QWP_Mcd" 
    ROOFTOP_IDS = ["ce99-366c-7b6e-cb6a"] 

    solcast = SolcastWeatherForecast(api_key=API_KEY, rooftop_ids=ROOFTOP_IDS, forecast_hours=48)

    # Fetch forecast (using cache if available)
    forecast_data = solcast.get_forecast(use_cache=True)

    # Display data
    print(forecast_data)
