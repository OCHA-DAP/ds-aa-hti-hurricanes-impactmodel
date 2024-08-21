#!/usr/bin/env python3
import importlib
import os
import sys
from pathlib import Path

import pandas as pd

from src.utils import blob, grid

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

import ast


def complete_impact_data(df_impact_grid):
    # Step 1: Get all unique grid points and total_pop from df_aux
    unique_grid_points = (
        df_impact_grid[["grid_point_id", "total_pop"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Step 2: Get all combinations of typhoon_name, typhoon_year, and grid_point_id
    unique_typhoons = (
        df_impact_grid[["typhoon_name", "typhoon_year"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Step 3: For each unique typhoon event, create a full set of grid points
    all_combinations = unique_typhoons.merge(unique_grid_points, how="cross")

    # Step 4: Merge with the original df_impact_grid to add missing grid points
    df_complete = all_combinations.merge(
        df_impact_grid,
        on=["typhoon_name", "typhoon_year", "grid_point_id", "total_pop"],
        how="left",
    )

    # Step 5: Fill missing perc_affected_pop_grid_grid with 0
    df_complete["perc_affected_pop_grid_grid"].fillna(0, inplace=True)
    df_complete["affected_pop"] = True

    # Step 6: Non impacting events
    wind_dir = (
        PROJECT_PREFIX + "/windfield/output_dir/windfield_data_hti_overlap.csv"
    )
    df_typhoons = blob.load_csv(wind_dir)
    df_typhoons = df_typhoons[
        ["typhoon_name", "typhoon_year", "affected_pop"]
    ].drop_duplicates()
    df_typhoons_nodmg = df_typhoons[
        df_typhoons.affected_pop == False
    ].reset_index(drop=True)

    df_dmg = df_impact_grid.copy()
    df_aux = df_dmg[["grid_point_id", "total_pop"]].drop_duplicates()
    grid_cells = df_aux.grid_point_id.tolist()
    total_pop = df_aux.total_pop.tolist()

    df_typhoons_nodmg["grid_point_id"] = str(grid_cells)
    df_typhoons_nodmg["total_pop"] = str(total_pop)
    df_typhoons_nodmg["grid_point_id"] = (
        df_typhoons_nodmg["grid_point_id"]
        .astype("str")
        .apply(ast.literal_eval)
    )

    df_typhoons_nodmg["total_pop"] = (
        df_typhoons_nodmg["total_pop"].astype("str").apply(ast.literal_eval)
    )
    df_typhoons_nodmg_exploded = df_typhoons_nodmg.explode(
        ["grid_point_id", "total_pop"]
    )
    df_impact_complete = (
        pd.concat([df_typhoons_nodmg_exploded, df_complete])
        .fillna(0)
        .reset_index(drop=True)
    )
    return df_impact_complete


def create_dynamic_features(cell_size, rain_dir):
    # Create grid cells with custom size
    (
        grid_all,
        grid_land_overlap,
        grid_centroids_all,
        grid_land_overlap_centroids,
        grid_muni,
    ) = grid.create_grid(cell_size=cell_size, save_to_blob=False)

    # Load wind data
    sys.path.append(os.path.abspath("src/datasources/01-IbTracks"))
    wind = importlib.import_module("wind_to_grid")

    # Load impact data and process storm tracks
    all_events, non_impacting_events = wind.load_impact_data()
    all_events["typhoon_name"] = all_events["typhoon_name"].str.strip()
    tc_tracks = wind.get_storm_tracks(all_events=all_events)
    tracks = wind.proccess_storm_tracks(tc_tracks=tc_tracks)

    df_windfield = wind.create_windfield_features(
        tracks=tracks,
        non_impacting_events=non_impacting_events,
        gdf=grid_land_overlap_centroids,
        gdf_all=grid_centroids_all,
        save_to_blob=False,
    )

    # Load rain data
    sys.path.append(os.path.abspath("src/datasources/02-NASA PPS"))
    rain = importlib.import_module("rainfall_dataset")

    typhoon_mean, typhoon_max = rain.create_rainfall_dataset(
        grid=grid_land_overlap,
        save_to_blob=False,
        load_from_blob=False,
        local_path=rain_dir,
    )

    df_meta = rain.load_metadata()
    df_rainfall_mean, df_rainfall_max = rain.compute_stats(
        load_from_blob=False,
        save_to_blob=False,
        typhoon_mean=typhoon_mean,
        typhoon_max=typhoon_max,
        stat_list=["mean"],
    )

    # Load population and impact data
    sys.path.append(os.path.abspath("src/datasources/03-Meta Data for Good"))
    population = importlib.import_module("population_dataset")
    impact = importlib.import_module("grid_impact_based_on_pop_data")

    grid_pop_df = population.get_population_data(
        grid=grid_land_overlap, save_to_blob=False
    )

    df_impact = blob.load_emdat()
    df_impact["typhoon_name"] = df_impact["typhoon_name"].str.strip()

    ids_mun = grid_muni.copy().reset_index(drop=True)

    impact_data_grid_no_weather = impact.impact_to_grid(
        grid_pop_df=grid_pop_df,
        ids_mun=ids_mun,
        df_impact=df_impact,
        save_to_blob=False,
    )

    df_impact_grid = impact_data_grid_no_weather[
        [
            "typhoon_name",
            "Year",
            "id",
            "total_pop",
            "perc_affected_pop_grid_grid",
        ]
    ]
    df_impact_grid = df_impact_grid.rename(
        {"id": "grid_point_id", "Year": "typhoon_year"}, axis=1
    )

    df_impact_complete = complete_impact_data(df_impact_grid)

    # Merge features
    df_wind_impact = df_windfield.drop("geometry", axis=1).merge(
        df_impact_complete
    )
    df_wind_impact["typhoon"] = df_wind_impact[
        "typhoon_name"
    ] + df_wind_impact["typhoon_year"].astype(str)
    df_rainfall_mean = df_rainfall_mean.rename({"id": "grid_point_id"}, axis=1)
    df_dynamic = df_wind_impact.merge(df_rainfall_mean)

    return df_dynamic


if __name__ == "__main__":
    # Local path to rain data
    rain_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/input/gpm_data/rainfall_data/output_hhr"
    )
    # Cell size
    cell_size = 0.5
    df_dynamic = create_dynamic_features(
        cell_size=cell_size, rain_dir=rain_dir
    )
    # Save to blob
    blob_name = (
        f"{PROJECT_PREFIX}/GRID_CELL_SIZE/{cell_size}/dynamic_features.csv"
    )
    csv_data = df_dynamic.to_csv(index=False)
    blob.upload_blob_data(blob_name=blob_name, data=csv_data)
