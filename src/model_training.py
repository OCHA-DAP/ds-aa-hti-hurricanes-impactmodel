#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from xgboost.sklearn import XGBRegressor

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"


def get_training_dataset(weather_constraints=True):
    if weather_constraints:
        blob_dir = (
            PROJECT_PREFIX
            + "/features_combined/training_dataset_hti_with_weather_thres.csv"
        )
    else:
        blob_dir = (
            PROJECT_PREFIX
            + "/features_combined/training_dataset_hti_without_weather_thres.csv"
        )

    return blob.load_csv(blob_dir)


def two_stage_model(
    df_hti,
    features,
    thres,
    weather_constraints=True,
    project_prefix=PROJECT_PREFIX,
    prod_dev="dev",
):
    # Split X and y from dataframe features
    X = df_hti[features]
    y = df_hti["perc_aff_pop_grid"]

    # Step 1: Train the initial XGBRegressor
    xgb = XGBRegressor(
        booster="gbtree",
        n_estimators=100,
        eval_metric=["rmse", "logloss"],
        verbosity=0,
    )
    xgb.fit(X, y)

    # Step 2: Train the XGBClassifier for binary classification
    y_bin = (y >= thres) * 1
    xgb_class = XGBClassifier(eval_metric=["error", "logloss"])
    xgb_class.fit(X, y_bin)

    # Create Reduced dataset (basically X + y)
    reduced_df = X.copy()
    reduced_df["perc_aff_pop_grid"] = y.values
    reduced_df["predicted_value"] = xgb_class.predict(X)

    # Just Predicted Damage dataset
    filtered_df = reduced_df[reduced_df.predicted_value == 1]

    # Step 3: Train the XGBRegressor on the reduced dataset
    X_r = filtered_df[features]
    y_r = filtered_df["perc_aff_pop_grid"]
    xgbR = XGBRegressor(
        booster="gbtree",
        n_estimators=100,
        eval_metric=["rmse", "logloss"],
        verbosity=0,
    )
    xgbR.fit(X_r, y_r)

    # Save models to temporary files and upload to blob storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_save_path = Path(tmp_dir)

        # Save models locally
        xgb_path = model_save_path / "xgb.pkl"
        xgb_class_path = model_save_path / "xgb_class.pkl"
        xgbR_path = model_save_path / "xgbR.pkl"
        joblib.dump(xgb, xgb_path)
        joblib.dump(xgb_class, xgb_class_path)
        joblib.dump(xgbR, xgbR_path)

        # Define blob storage paths
        if weather_constraints:
            xgb_blob_path = (
                f"{project_prefix}/model/weather_constraints/xgb.pkl"
            )
            xgb_class_blob_path = (
                f"{project_prefix}/model/weather_constraints/xgb_class.pkl"
            )
            xgbR_blob_path = (
                f"{project_prefix}/model/weather_constraints/xgbR.pkl"
            )
        else:
            xgb_blob_path = (
                f"{project_prefix}/model/no_weather_constraints/xgb.pkl"
            )
            xgb_class_blob_path = (
                f"{project_prefix}/model/no_weather_constraints/xgb_class.pkl"
            )
            xgbR_blob_path = (
                f"{project_prefix}/model/no_weather_constraints/xgbR.pkl"
            )

        # Upload to blob storage
        blob.upload_blob_data(
            xgb_blob_path, xgb_path.read_bytes(), prod_dev=prod_dev
        )
        blob.upload_blob_data(
            xgb_class_blob_path, xgb_class_path.read_bytes(), prod_dev=prod_dev
        )
        blob.upload_blob_data(
            xgbR_blob_path, xgbR_path.read_bytes(), prod_dev=prod_dev
        )

    return xgb, xgb_class, xgbR


def run_model(
    thres,
    project_prefix=PROJECT_PREFIX,
    weather_constraints=True,
    prod_dev="dev",
):
    # Load HTI training dataset
    df_hti = get_training_dataset(weather_constraints=weather_constraints)
    # Rename target variable
    df_hti = df_hti.rename(
        {
            "perc_affected_pop_grid_grid": "perc_aff_pop_grid",
            "mean_altitude": "mean_elev",
        },
        axis=1,
    )

    # Features
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

    # Train and save models
    two_stage_model(
        df_hti=df_hti,
        features=features,
        weather_constraints=weather_constraints,
        thres=thres,
        project_prefix=project_prefix,
    )


if __name__ == "__main__":
    # Save models with weather constraints
    run_model(thres=5, weather_constraints=True)

    # Save models without weather constraints
    run_model(thres=1, weather_constraints=False)
