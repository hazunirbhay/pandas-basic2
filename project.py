import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import streamlit as st


def fetch_weather_data(lat=28.6139, lon=77.2090, days=90):
  """Fetch 90 days of historical hourly weather data (default: New Delhi)."""
  print("Fetching historical weather observations from Open-Meteo...")
  url = "https://archive-api.open-meteo.com/v1/archive"

  # Offset end date slightly to ensure verified archive data availability
  end_date = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
  start_date = (
      pd.Timestamp.now() - pd.Timedelta(days=days + 2)
  ).strftime("%Y-%m-%d")

  params = {
      "latitude": lat,
      "longitude": lon,
      "start_date": start_date,
      "end_date": end_date,
      "hourly": (
          "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
      ),
      "timezone": "auto",
  }

  response = requests.get(url, params=params).json()

  df = pd.DataFrame({
      "timestamp": pd.to_datetime(response["hourly"]["time"]),
      "temp": response["hourly"]["temperature_2m"],
      "humidity": response["hourly"]["relative_humidity_2m"],
      "pressure": response["hourly"]["surface_pressure"],
      "wind_speed": response["hourly"]["wind_speed_10m"],
  })
  return df.dropna()


def engineer_features(df, forecast_horizon=6):
  """Generate cyclic temporal features, historical lags, and future targets."""
  df = df.copy()

  # Temporal cycles
  df["hour"] = df["timestamp"].dt.hour
  df["day_of_year"] = df["timestamp"].dt.dayofyear

  # Historical lookback lags
  for lag in [1, 3, 6, 24]:
    df[f"temp_lag_{lag}"] = df["temp"].shift(lag)
    df[f"pressure_lag_{lag}"] = df["pressure"].shift(lag)

  # Target variable: temperature N hours into the future
  df["target_temp"] = df["temp"].shift(-forecast_horizon)

  return df.dropna()


def run_pipeline():
  # 1. Pipeline Stage: Ingestion
  raw_df = fetch_weather_data()

  # 2. Pipeline Stage: Feature Engineering
  processed_df = engineer_features(raw_df, forecast_horizon=6)

  feature_cols = [
      c
      for c in processed_df.columns
      if c not in ["timestamp", "target_temp"]
  ]
  X = processed_df[feature_cols]
  y = processed_df["target_temp"]

  # 3. Pipeline Stage: Time-Series Chronological Split (80% Train, 20% Test)
  split_idx = int(len(X) * 0.8)
  X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
  y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

  # 4. Pipeline Stage: Model Training
  print("Training Random Forest Regressor model...")
  model = RandomForestRegressor(
      n_estimators=100, max_depth=12, random_state=42
  )
  model.fit(X_train, y_train)

  # 5. Pipeline Stage: Evaluation
  predictions = model.predict(X_test)
  mae = mean_absolute_error(y_test, predictions)
  rmse = root_mean_squared_error(y_test, predictions)

  print("\n--- Model Evaluation ---")
  print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
  print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")

  # 6. Visualizing Predictions vs Reality
  plt.figure(figsize=(12, 5))
  plt.plot(
      processed_df["timestamp"].iloc[split_idx:],
      y_test.values,
      label="Actual Temp (°C)",
      alpha=0.75,
  )
  plt.plot(
      processed_df["timestamp"].iloc[split_idx:],
      predictions,
      label="6-Hour Ahead Prediction (°C)",
      linestyle="--",
      color="orange",
  )
  plt.title("AI Weather Model: 6-Hour Forecast vs Ground Truth")
  plt.xlabel("Timestamp")
  plt.ylabel("Temperature (°C)")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  st.pyplot(plt.gcf())


if __name__ == "__main__":
  st.title("AI weather predictor")
  run_pipeline()
  st.pyplot(plt.gcf())