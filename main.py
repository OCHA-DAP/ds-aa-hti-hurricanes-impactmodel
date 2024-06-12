#!/usr/bin/env python3
from datetime import datetime
from io import BytesIO

import joblib
import pandas as pd

from src.datasources.ECMWF import realtime_wind_data as wind
from src.datasources.GEFS import realtime_rainfall_data as rain
from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

# Features of the model
features = [
    "wind_speed",
    "track_distance",
    "total_buildings",
    "total_pop",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "coast_length",
    "with_coast",
    "mean_elev",
    "mean_slope",
    "mean_rug",
    "IWI",
]


def create_input_dataset():
    # Load stationary features
    stationary_dir = (
        PROJECT_PREFIX + "/features_combined/stationary_data_hti.csv"
    )
    df_stationary = blob.load_csv(stationary_dir).drop_duplicates()

    # Load realtime forecasts
    today = datetime.now().strftime("%Y%m%d")
    df_windfield = wind.load_windspeed_data(date=today)
    df_rainfall = rain.load_rainfall_data(date=today)
    df_forecast = df_rainfall.merge(
        df_windfield, left_on="id", right_on="grid_point_id"
    )

    # Merge features
    df_input = df_stationary.merge(df_forecast)
    return df_input


def predict_new_data(
    new_data,
    features=features,
    project_prefix=PROJECT_PREFIX,
    prod_dev="dev",
    thres=0.5,
    weather_constraints=True,
):
    # Define model base path based on weather constraints
    if weather_constraints:
        model_base_path = f"{project_prefix}/model/weather_constraints"
    else:
        model_base_path = f"{project_prefix}/model/no_weather_constraints"

    # Define blob paths for models
    xgb_blob_path = f"{model_base_path}/xgb.pkl"
    xgb_class_blob_path = f"{model_base_path}/xgb_class.pkl"
    xgbR_blob_path = f"{model_base_path}/xgbR.pkl"

    # Load models from blob storage
    xgb_data = blob.load_blob_data(xgb_blob_path, prod_dev=prod_dev)
    xgb_class_data = blob.load_blob_data(
        xgb_class_blob_path, prod_dev=prod_dev
    )
    xgbR_data = blob.load_blob_data(xgbR_blob_path, prod_dev=prod_dev)

    # Load models using joblib
    xgb = joblib.load(BytesIO(xgb_data))
    xgb_class = joblib.load(BytesIO(xgb_class_data))
    xgbR = joblib.load(BytesIO(xgbR_data))

    # Split X from new_data
    X_new = new_data[features]

    # Step 1: Predict initial regression values
    y_pred_xgb = xgb.predict(X_new)

    # Step 2: Predict binary classification values
    y_pred_class = xgb_class.predict(X_new)

    # Create Reduced dataset for new data
    new_data["predicted_value"] = y_pred_class

    # Just Predicted Damage dataset
    filtered_new_data = new_data[new_data.predicted_value == 1]

    # Step 3: Predict regression values on the reduced dataset
    X_r_new = filtered_new_data[features]
    y_pred_r = xgbR.predict(X_r_new)

    # Create final predictions
    filtered_new_data["predicted_percent_aff"] = y_pred_r

    not_filtered_new_data = new_data[new_data.predicted_value == 0]
    not_filtered_new_data["predicted_percent_aff"] = xgb.predict(
        not_filtered_new_data[features]
    )
    try:
        # Try concatenate results for non affected and affected grid cells (if both exist)
        final_predictions = pd.concat(
            [not_filtered_new_data, filtered_new_data]
        )
    except:
        if len(filtered_new_data) > 1:
            # For events with not no dmg predictions
            final_predictions = filtered_new_data
        else:
            # For events with no dmg predictions
            final_predictions = not_filtered_new_data

    return final_predictions


if __name__ == "__main__":
    # For folder and files naming
    today = datetime.now().strftime("%Y%m%d")
    # Check high windspeed in the Haiti surroundings (+/- degree)
    try:
        trigger = wind.create_windfield_dataset(thres=120, deg=3)
        if trigger:
            # Input dataset
            df_input = create_input_dataset()
            # Predict on trained model
            df_pred = predict_new_data(new_data=df_input)

            # Save predictions
            csv_data = df_pred.to_csv(index=False)
            csv_dir = (
                PROJECT_PREFIX
                + "/model/predictions/impact_predictions_{}.csv".format(today)
            )
            blob.upload_blob_data(csv_dir, csv_data)

            # Create a TXT file with a message
            with open(f"output_{today}.txt", "w") as txt_file:
                txt_file.write(
                    f"Trigger activatedt. Check blob storage for output.\nDate={today}"
                )
        else:
            # Create a TXT file with a message
            with open(f"output_{today}.txt", "w") as txt_file:
                txt_file.write(
                    f"Trigger condition not met. No output generated.\nDate={today}"
                )
    except ValueError as e:
        # Create a TXT file with a message
        with open(f"output_{today}.txt", "w") as txt_file:
            txt_file.write(
                f"ECMWF not responding. Try again later.\nDate={today}"
            )
