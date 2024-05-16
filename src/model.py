#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from xgboost.sklearn import XGBRegressor

# Directories
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/04_model_output_dataset"
)
output_dir.mkdir(exist_ok=True)


def get_training_dataset(weather_constraints=True):
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/03_model_input_dataset"
    )

    if weather_constraints:
        filename = input_dir / "training_dataset_hti_with_weather_thres.csv"
    else:
        filename = input_dir / "training_dataset_hti_without_weather_thres.csv"

    return pd.read_csv(filename)


def two_stage_model(df_hti, features, thres):
    # Initialize variables
    y_test_typhoon_combined = []
    y_pred_typhoon_combined = []
    rmse_combined = []

    hti_aux = df_hti[["typhoon_name", "typhoon_year"]].drop_duplicates()
    # LOOCV
    for typhoon, year in zip(hti_aux["typhoon_name"], hti_aux["typhoon_year"]):
        # print(typhoon)
        """PART 1: Train/Test"""

        # LOOCV
        df_test = df_hti[
            (df_hti["typhoon_name"] == typhoon)
            & (df_hti["typhoon_year"] == year)
        ]  # Test set: HTI event
        df_train = df_hti[
            (df_hti["typhoon_name"] != typhoon)
            & (df_hti["typhoon_year"] != year)
        ]  # Train set: everything else

        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train["perc_aff_pop_grid"]
        y_test = df_test["perc_aff_pop_grid"]

        """ PART 2: XGB Regressor """

        # XGBRegressor
        xgb = XGBRegressor(
            booster="gbtree",
            n_estimators=100,
            # max_depth=5,
            eval_metric=["rmse", "logloss"],
            verbosity=0,
        )

        eval_set = [(X_train, y_train)]
        xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Make prediction on train and test data
        y_pred_train_xgb = xgb.predict(X_train)
        y_pred_test_xgb = xgb.predict(X_test)

        """ PART 3: CLssifier  """

        # Define a threshold to separate target into damaged and not_damaged
        y_test_bool = y_test >= thres
        y_train_bool = y_train >= thres
        y_test_bin = (y_test_bool) * 1  # nice way to convert boolean to binary
        y_train_bin = (y_train_bool) * 1

        # Use XGBClassifier as a Machine Learning model to fit the data
        xgb_model = XGBClassifier(eval_metric=["error", "logloss"])
        eval_set = [(X_test, y_test_bin)]
        xgb_model.fit(X_train, y_train_bin, eval_set=eval_set, verbose=False)

        # Make prediction on test data
        y_pred_test = xgb_model.predict(X_test)
        # Make prediction on train data
        y_pred_train = xgb_model.predict(X_train)

        # Create Reduced dataset (basically X_reduced = X_train + y_train)
        reduced_df = X_train.copy()
        reduced_df["perc_aff_pop_grid"] = y_train.values
        reduced_df["predicted_value"] = y_pred_train

        # Just Predicted Damage dataset
        flitered_df = reduced_df[reduced_df.predicted_value == 1]

        """ PART 4: Regression  """

        ### Third step is to train XGBoost regression model for this reduced train data (including damg>10.0%)

        # Split X and y from dataframe features (JUST WHEN predicted_value == 1, i.e fliterd_df)
        X_r = flitered_df[features]
        y_r = flitered_df["perc_aff_pop_grid"]

        # XGBoost Reduced Overfitting
        xgbR = XGBRegressor(
            booster="gbtree",
            n_estimators=100,
            # max_depth=5,
            eval_metric=["rmse", "logloss"],
            verbosity=0,
        )

        eval_set = [(X_r, y_r)]
        xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)

        # Make prediction on train and global test data
        y_pred_r = xgbR.predict(X_r)
        y_pred_test_total = xgbR.predict(X_test)

        # Calculate RMSE in total
        mse_train_idxR = mean_squared_error(y_r, y_pred_r)
        rmse_trainR = np.sqrt(mse_train_idxR)

        mse_idxR = mean_squared_error(y_test, y_pred_test_total)
        rmseR = np.sqrt(mse_idxR)

        """ PART 5: All together      """

        ## Last step is to add model combination (model M1 with model MR)
        # Check the result of classifier for TEST SET
        reduced_test_df = X_test.copy()

        # joined X_test with countinous target and binary predicted values with Classificator XGB (step 2)
        reduced_test_df["perc_aff_pop_grid"] = y_test.values
        reduced_test_df["predicted_value"] = y_pred_test

        # damaged prediction
        fliterd_test_df1 = reduced_test_df[
            reduced_test_df.predicted_value == 1
        ]
        # not damaged prediction
        fliterd_test_df0 = reduced_test_df[
            reduced_test_df.predicted_value == 0
        ]

        # keep only the features
        X1 = fliterd_test_df1[features]  # just damage
        X0 = fliterd_test_df0[features]  # not damage

        # For the output equal to 1 apply XGBReg (step 3) to evaluate the performance
        y1_pred = xgbR.predict(X1)
        y1 = fliterd_test_df1["perc_aff_pop_grid"]

        # For the output equal to 0 apply XGBReg (M1-Model -510 model- of step 1) to evaluate the performance
        y0_pred = xgb.predict(X0)
        y0 = fliterd_test_df0["perc_aff_pop_grid"]

        fliterd_test_df0["predicted_percent_aff"] = y0_pred
        fliterd_test_df1["predicted_percent_aff"] = y1_pred

        # Join two dataframes together
        try:
            join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])
        except:
            if len(fliterd_test_df1) > 1:
                join_test_dfs = fliterd_test_df1
            else:
                # print(typhoon)
                join_test_dfs = (
                    fliterd_test_df0  # For events with no dmg predictions
                )

        """ PART 6: Calculate Final Stuff  """

        # Calculate RMSE in total
        mse_combined_model = mean_squared_error(
            join_test_dfs["perc_aff_pop_grid"],
            join_test_dfs["predicted_percent_aff"],
        )
        rmse_combined_model = np.sqrt(mse_combined_model)
        rmse_combined.append(rmse_combined_model)

        y_join = join_test_dfs["perc_aff_pop_grid"]
        y_pred_join = join_test_dfs["predicted_percent_aff"]

        y_test_typhoon_combined.append(y_join)
        y_pred_typhoon_combined.append(y_pred_join)

    return y_test_typhoon_combined, y_pred_typhoon_combined, rmse_combined


def run_model(thres, weather_constraints=True):
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

    # 2-STG-XGB
    (
        y_test_typhoon_2stg,
        y_pred_typhoon_2stg,
        rmse_total_2stg,
    ) = two_stage_model(
        df_hti=df_hti,
        features=features,
        thres=thres,  # Damage threshold by grid cell
    )

    return y_test_typhoon_2stg, y_pred_typhoon_2stg, rmse_total_2stg


def process_training_dataset(weather_constraints=True):
    # Get training dataset based on weather constraints
    df_hti = get_training_dataset(weather_constraints)

    # Extract relevant columns and drop duplicates
    df_aux = df_hti[
        ["typhoon_name", "typhoon_year", "affected_pop"]
    ].drop_duplicates()
    df_aux = (
        df_aux.rename(columns={"typhoon_year": "Year"})
        .dropna()
        .reset_index(drop=True)
    )

    # Initialize output DataFrame
    df_out = pd.DataFrame()

    # Iterate over typhoons and years
    for name, year in zip(df_aux.typhoon_name, df_aux.Year):
        # Select typhoon/event
        idx = df_aux[
            (df_aux.typhoon_name == name) & (df_aux.Year == year)
        ].index[0]
        df_typhoon = df_hti[
            (df_hti.typhoon_name == name) & (df_hti.typhoon_year == year)
        ]
        y_pred = (
            y_pred_typhoon_weather[idx]
            if weather_constraints
            else y_pred_typhoon_no_weather[idx]
        )

        # Add feature "predictive_damage"
        df_typhoon["predicted_damage"] = y_pred

        # Force predicted values into the [0,100] range
        df_typhoon.loc[df_typhoon.predicted_damage < 0, "predicted_damage"] = 0
        df_typhoon.loc[
            df_typhoon.predicted_damage > 100, "predicted_damage"
        ] = 100

        # Concatenate output to df_out
        df_out = pd.concat([df_out, df_typhoon])

    return df_out


if __name__ == "__main__":
    # The threshold is for binarizing the impact on the second stage of the model (classifier)
    # This is a threshold at GRID LEVEL. I usually consider the quantile 75 of the distribution of impact
    # at grid level in the subset of just impacting events

    # Model with weather constraints in grid disggregation stage
    (
        y_test_typhoon_weather,
        y_pred_typhoon_weather,
        rmse_total_weather,
    ) = run_model(thres=5, weather_constraints=True)
    # Model without weather constraints in grid disggregation stage
    (
        y_test_typhoon_no_weather,
        y_pred_typhoon_no_weather,
        rmse_total_no_weather,
    ) = run_model(thres=1, weather_constraints=False)

    """      Save outputs       """
    # Weather constraints approach
    df_out_weather = process_training_dataset(
        weather_constraints=True
    ).reset_index(drop=True)
    df_out_weather.to_csv(
        output_dir / "features_plus_predictions_weather_thres.csv"
    )
    # No Weather constraints approach
    df_out_no_weather = process_training_dataset(
        weather_constraints=False
    ).reset_index(drop=True)
    df_out_no_weather.to_csv(
        output_dir / "features_plus_predictions_no_weather_thres.csv"
    )
