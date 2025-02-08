import pandas as pd
import numpy as np
import pickle
import logging
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle
import logging
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # XGBoost supports missing values
from sklearn.metrics import r2_score

class MLForecaster:
    """
    A Machine Learning-based time-series forecaster using autoregressive features.
    """

    def __init__(self, data: pd.DataFrame, num_lags: int, model_type: str = "LinearRegression"):
        """
        Initialize the forecaster.

        :param data: DataFrame with a DateTimeIndex containing historic load data.
        :param num_lags: Number of autoregressive lags to use.
        :param model_type: Machine Learning model to use ('LinearRegression', 'ElasticNet', 'KNeighborsRegressor', 'RandomForest', 'XGBoost').
        """
        if "load" not in data.columns:
            raise ValueError("Data must contain a 'load' column.")

        self.data = data.copy()
        self.num_lags = num_lags
        self.model_type = model_type
        self.forecaster = None

        # Ensure datetime index
        self.data.index = pd.to_datetime(self.data.index)
        self.data.index = self.data.index.tz_localize("UTC") if self.data.index.tz is None else self.data.index.tz_convert("UTC")

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds time-based features to the dataset.

        :param df: DataFrame with a DateTimeIndex.
        :return: DataFrame with added time features.
        """
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        return df

    def train(self):
        """
        Trains the machine learning model while allowing missing values.
        """
        print(f"Training model: {self.model_type}")

        # Add time-based features
        self.data = self.add_time_features(self.data)

        # Prepare dataset
        X = self.data.drop(columns=["load"])
        y = self.data["load"]

        # Drop rows where `y` is NaN (for models that require it)
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Select model
        model_map = {
            "LinearRegression": LinearRegression(),
            "ElasticNet": ElasticNet(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(missing=np.nan),  # Handles missing values natively
        }
        base_model = model_map.get(self.model_type, LinearRegression())

        # Initialize and train forecaster
        self.forecaster = ForecasterRecursive(regressor=base_model, lags=self.num_lags)
        self.forecaster.fit(y=y, exog=X)

        print(" Model trained successfully.")

    def predict(self, forecast_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate forecasts from the trained model.

        :param forecast_dates: Future dates for which predictions are needed.
        :return: Forecasted load values.
        """
        if self.forecaster is None:
            raise ValueError("Model must be trained before predicting.")

        # Ensure exogenous features match forecast dates
        exog_future = self.add_time_features(pd.DataFrame(index=forecast_dates))

# Convert exog_future to a RangeIndex
        exog_future.index = forecast_dates
        print(f" Last training timestamp: {self.data.index[-1]}")
        print(f" First forecast timestamp: {forecast_dates[0]}")  # Should match last_window + 30min

        # Generate predictions
        predictions = self.forecaster.predict(steps=len(forecast_dates), exog=exog_future)
        return predictions

    def save_model(self, file_path: str):
        """
        Saves the trained model.

        :param file_path: Path to save the model.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self.forecaster, file)
        print(f" Model saved to {file_path}")

    def load_model(self, file_path: str):
        """
        Loads a pre-trained model.

        :param file_path: Path to the saved model.
        """
        with open(file_path, "rb") as file:
            self.forecaster = pickle.load(file)
        print(f" Model loaded from {file_path}")

if __name__ == "__main__":
    # Load 30-minute interval training data
    csv_path = "training_load_data_shifted.csv"
    df_load = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    if "load" not in df_load.columns:
        raise ValueError("CSV file must contain a 'load' column.")

    # Ensure timestamps are properly formatted
    df_load.index = pd.to_datetime(df_load.index)
    df_load.index = df_load.index.tz_localize("UTC") if df_load.index.tz is None else df_load.index.tz_convert("UTC")
    df_load = df_load.asfreq("30min")  # Ensure consistent time frequency

    # Check and handle missing values
    missing_values = df_load["load"].isnull().sum()
    if missing_values > 0:
        print(f" Warning: Found {missing_values} missing values in 'load' column. Filling using forward fill (ffill).")
        df_load["load"] = df_load["load"].fillna(method="ffill").fillna(method="bfill")

    # Ensure forecast starts right after the last training timestamp
    last_timestamp = df_load.index[-1]  # Example: 2025-02-07 07:30:00+00:00
    forecast_dates = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), periods=48, freq="30min")
    forecast_dates.freq = "30min"  # Explicitly set frequency

    # Print verification logs
    print(f" Last training timestamp: {last_timestamp}")
    print(f" First forecast timestamp: {forecast_dates[0]}")  
    # Initialize and train the ML forecaster (Use 'XGBoost' or 'RandomForest' to handle missing values better)
    forecaster = MLForecaster(df_load, num_lags=48, model_type="XGBoost")
    forecaster.train()

    # Save and reload model
    model_file = "trained_ml_model.pkl"
    forecaster.save_model(model_file)
    forecaster.load_model(model_file)

    # Generate predictions
    forecast = forecaster.predict(forecast_dates)
    forecast.to_csv("ml_forecast.csv")

    print("\n Forecasted Load:\n", forecast.head())