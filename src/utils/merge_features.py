#!/usr/bin/env python3
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Directories
input_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features"
)
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
)
output_dir.mkdir(exist_ok=True)


def get_bld_data():
    # Read file
    filename = input_dir / "02_housing_damage/input"
    bld_by_grid = pd.read_csv(filename / "hti_google_bld_grid_count.csv")
    return bld_by_grid


def get_impact_data(weather_constraints=True):
    # Non impacting events
    filename_events = input_dir / "01_windfield/windfield_data_hti_overlap.csv"
    df_typhoons = pd.read_csv(filename_events)
    df_typhoons = df_typhoons[
        ["typhoon_name", "typhoon_year", "affected_pop"]
    ].drop_duplicates()
    df_typhoons_nodmg = df_typhoons[
        df_typhoons.affected_pop == False
    ].reset_index(drop=True)

    # Impacting events
    if weather_constraints:
        filename = (
            input_dir
            / "02_housing_damage/output/impact_data_grid_step_disaggregation_weather.csv"
        )
    else:
        filename = (
            input_dir
            / "02_housing_damage/output/impact_data_grid_step_disaggregation_no_weather.csv"
        )

    df_dmg = pd.read_csv(filename)

    # Add bld data
    df_bld = get_bld_data()
    df_dmg = df_dmg.merge(df_bld, on="id")

    # Rename some columns
    df_dmg = df_dmg.rename(
        {
            "id": "grid_point_id",
            "affected_population": "total_pop_affected",
            "count": "total_buildings",
        },
        axis=1,
    )
    df_dmg["typhoon_name"] = df_dmg["typhoon_name"].str.upper()
    df_dmg.Year = df_dmg.Year.astype("int64")

    # Add information to non impacting events
    df_aux = df_dmg[
        ["grid_point_id", "total_buildings", "total_pop"]
    ].drop_duplicates()
    grid_cells = df_aux.grid_point_id.tolist()
    total_bld = df_aux.total_buildings.tolist()
    total_pop = df_aux.total_pop.tolist()

    df_typhoons_nodmg["grid_point_id"] = str(grid_cells)
    df_typhoons_nodmg["total_buildings"] = str(total_bld)
    df_typhoons_nodmg["total_pop"] = str(total_pop)
    df_typhoons_nodmg["grid_point_id"] = (
        df_typhoons_nodmg["grid_point_id"]
        .astype("str")
        .apply(ast.literal_eval)
    )
    df_typhoons_nodmg["total_buildings"] = (
        df_typhoons_nodmg["total_buildings"]
        .astype("str")
        .apply(ast.literal_eval)
    )
    df_typhoons_nodmg["total_pop"] = (
        df_typhoons_nodmg["total_pop"].astype("str").apply(ast.literal_eval)
    )
    df_typhoons_nodmg_exploded = df_typhoons_nodmg.explode(
        ["grid_point_id", "total_buildings", "total_pop"]
    )
    df_typhoons_nodmg_exploded = df_typhoons_nodmg_exploded.rename(
        {"typhoon_year": "Year"}, axis=1
    )

    # Merge non impacting events and impacting ones
    df_damage_all = pd.concat([df_dmg, df_typhoons_nodmg_exploded]).fillna(0)
    return df_damage_all


def get_wind_data():
    # Read file
    filename = input_dir / "01_windfield/windfield_data_hti_overlap.csv"
    df_windfield = pd.read_csv(filename)
    df_windfield = df_windfield.drop("geometry", axis=1)
    df_windfield = df_windfield.drop_duplicates()
    return df_windfield


def get_rainfall_data():
    # Read file
    filename = input_dir / "03_rainfall/output/rainfall_data_rw_mean.csv"
    df_rainfall = pd.read_csv(filename)
    # Clean csv
    df_rainfall[["typhoon_name", "typhoon_year"]] = df_rainfall[
        "typhoon"
    ].str.split("(\d+)", expand=True)[[0, 1]]
    df_rainfall["typhoon_name"] = df_rainfall["typhoon_name"].str.upper()
    df_rainfall["typhoon_year"] = df_rainfall["typhoon_year"].astype(int)
    df_rainfall = df_rainfall.rename(columns={"id": "grid_point_id"}).loc[
        :,
        [
            "typhoon_name",
            "typhoon_year",
            "grid_point_id",
            "rainfall_max_6h",
            "rainfall_max_24h",
        ],
    ]
    return df_rainfall


def get_IWI_data():
    filename_iwi = input_dir / "05_vulnerability/output/hti_iwi_bygrid_new.csv"
    df_iwi = pd.read_csv(filename_iwi)
    return df_iwi


def get_topo_data():
    filename_topo = (
        input_dir / "04_topography/output/topography_variables_bygrid.csv"
    )
    df_topo = pd.read_csv(filename_topo)
    df_topo = df_topo.rename({"id": "grid_point_id"}, axis=1)
    return df_topo


def merge_features(weather_constraints):
    df_damage_all = get_impact_data(weather_constraints=weather_constraints)
    df_windfield = get_wind_data()
    df_rainfall = get_rainfall_data()
    df_topo = get_topo_data()
    df_iwi = get_IWI_data()

    # Merge windfield and rainfall
    df_merge_wind_rain = df_windfield.merge(df_rainfall)

    # Merge with impact data
    df_merge_typhoons = (
        df_damage_all[
            [
                "typhoon_name",
                "Year",
                "grid_point_id",
                "total_pop_affected",
                "total_buildings",
                "total_pop",
                "perc_affected_pop_grid_grid",
            ]
        ]
        .rename({"Year": "typhoon_year"}, axis=1)
        .merge(df_merge_wind_rain)
    )

    # Merge with topography data
    df_merge_topo = df_merge_typhoons.merge(df_topo, on="grid_point_id")

    # Merge with vulnerability indexes
    df_all = df_merge_topo.merge(df_iwi, on="grid_point_id")

    # Common nans in topography and rainfall data
    df_all = df_all.fillna(0).reset_index(drop=True)
    return df_all


if __name__ == "__main__":
    # Features + target variable + weather constraints
    df_with_weather_cons = merge_features(weather_constraints=True)
    # Features + target variable + NO weather constraints
    df_without_weather_cons = merge_features(weather_constraints=False)
    # Create stationary dataset
    df_stationary = merge_features()
    features_stationary = [
        "grid_point_id",
        "IWI",
        "total_buildings",
        "total_pop",
        "with_coast",
        "coast_length",
        "mean_altitude",
        "mean_slope",
        "mean_rug",
    ]
    df_stationary = df_stationary[features_stationary]

    # To csv
    df_with_weather_cons.to_csv(
        output_dir / "training_dataset_hti_with_weather_thres.csv", index=False
    )
    df_without_weather_cons.to_csv(
        output_dir / "training_dataset_hti_without_weather_thres.csv",
        index=False,
    )
    df_stationary.to_csv(output_dir / "stationary_data_hti.csv", index=False)
