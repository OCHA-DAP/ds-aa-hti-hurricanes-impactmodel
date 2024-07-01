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

    # Load realtime wind forecasts
    today = datetime.now().strftime("%Y%m%d")
    df_windfield = wind.load_windspeed_data(date=today)

    # Load realtime rain forecasts
    event_metadata = rain.create_metadata(df_windfield)
    rain.create_rainfall_dataset(event_metadata=event_metadata)
    df_rainfall = rain.load_rainfall_data(date=today)
    
    # Merge wind and rainfall forecast data
    df_forecast = df_windfield.merge(df_rainfall, 
                    left_on=['unique_id', 'grid_point_id'],
                    right_on=['event', 'id'])

    # Merge all features
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

    # Add grid information
    final_predictions["grid_point_id"] = new_data["grid_point_id"]
    return final_predictions


# Aggregate predictions to adm1 level + add pre-defined bootstrapping error
def aggregate_predictions_adm1(final_predictions):
    # Load ids of municipalities
    ids_mun = blob.load_csv(
        PROJECT_PREFIX + "/grid/input_dir/grid_municipality_info.csv"
    )[["id", "ADM1_PCODE"]]

    # Actual number of people affected
    final_predictions["N_people_affected_predicted"] = (
        final_predictions["predicted_percent_aff"]
        * final_predictions["total_pop"]
        / 100
    )

    # If N people affected predicted is < 0 --> set it to 0
    final_predictions.loc[final_predictions["N_people_affected_predicted"] < 0, 'N_people_affected_predicted'] = 0

    # Merge with municipality info
    df_merged = final_predictions.merge(
        ids_mun, left_on="grid_point_id", right_on="id"
    )
    # For each storm, aggregate to ADM1
    df_adm1 = pd.DataFrame()
    for event in df_merged.unique_id.unique():

        df_event = df_merged[df_merged.unique_id == event]

        df_adm1_event = df_event.groupby("ADM1_PCODE").sum()[
            ["total_pop", "N_people_affected_predicted"]
        ].reset_index()

        df_adm1_event["perc_mun_affected"] = (
            100 * df_adm1_event["N_people_affected_predicted"] / df_adm1_event["total_pop"]
        )

        # Fix predictions (just in case, but it should work without this)
        df_adm1_event.loc[df_adm1_event["perc_mun_affected"] > 100, "perc_mun_affected"] = 100
        df_adm1_event.loc[df_adm1_event["perc_mun_affected"] < 0, "perc_mun_affected"] = 0
        
        # Concatenate all events
        df_adm1_event['unique_id'] = event
        df_adm1 = pd.concat([df_adm1, df_adm1_event])

    # Add Bootstrapping error
    # Pre-defined model-related bootstrapping error
    bin_errors = {"0 - 1": 0.03, "1 - 10": 1.44, "10 - 100": 4.80}

    # Function to determine bin error
    def get_bin_error(value):
        if 0 <= value < 1:
            return bin_errors["0 - 1"]
        elif 1 <= value < 10:
            return bin_errors["1 - 10"]
        elif 10 <= value <= 100:
            return bin_errors["10 - 100"]
        else:
            return None

    # Add Bootstrapping error
    df_adm1["model_error[%]"] = df_adm1["perc_mun_affected"].apply(
        get_bin_error
    )

    return df_adm1[
        [
            "unique_id",
            "ADM1_PCODE",
            "perc_mun_affected",
            "N_people_affected_predicted",
            "model_error[%]",
        ]
    ].reset_index(drop=True)


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
            # Aggregate predictions to ADM1 level
            df_pred_adm1 = aggregate_predictions_adm1(df_pred)

            # Save predictions
            csv_data = df_pred_adm1.to_csv(index=False)
            csv_dir = (
                PROJECT_PREFIX
                + "/model/predictions/impact_predictions_{}.csv".format(today)
            )
            blob.upload_blob_data(csv_dir, csv_data)

            # Create a TXT file with a message
            with open(f"output_{today}.txt", "w") as txt_file:
                txt_file.write(
                    f"Trigger activated. Check blob storage for output.\nDate={today}"
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
