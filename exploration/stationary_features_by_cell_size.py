#!/usr/bin/env python3
import importlib
import os
import sys

import geopandas as gpd

from src.utils import blob, grid

PROJECT_PREFIX = "ds-aa-hti-hurricanes"


def create_stationary_features(cell_size):
    # Create grid cells with custom size
    (
        grid_all,
        grid_land_overlap,
        grid_centroids_all,
        grid_land_overlap_centroids,
        grid_muni,
    ) = grid.create_grid(cell_size=cell_size, save_to_blob=False)

    # Load shapefile
    shp = blob.load_shp()
    shp = shp.to_crs("EPSG:4326")

    # Add the directory containing wind_to_grid.py and wind_functions.py to the system path
    sys.path.append(os.path.abspath("../src/datasources/04-DWTKNS SRTM"))

    # Import the wind_to_grid module for SRTM data
    srtm = importlib.import_module("topography_dataset")

    # Coast related features
    grid_coast = srtm.get_coast_features(grid=grid_land_overlap, shp=shp)

    # Topography features (ELEV, SLOPE, RUGG)
    df_terrain = srtm.get_topography_features(grid=grid_land_overlap)

    df_srtm = df_terrain.merge(grid_coast, on="id", how="left")
    df_srtm = df_srtm[
        [
            "id",
            "with_coast",
            "coast_length",
            "mean_elev",
            "mean_slope",
            "mean_rug",
        ]
    ]
    df_srtm = df_srtm.fillna(0)  # No coast length? Then it's 0
    df_srtm = df_srtm.rename({"id": "grid_point_id"}, axis=1)

    # Add the directory containing building data to the system path
    sys.path.append(
        os.path.abspath("../src/datasources/06-Google Open Buildings")
    )

    # Import the module for building data
    gob = importlib.import_module("buildings_by_grid")
    data_path = f"{PROJECT_PREFIX}/google/input_dir/google_footprint_data.csv"
    ggl_gdf = blob.load_csv(data_path)

    ggl_gdf_gpd = gpd.GeoDataFrame(
        ggl_gdf,
        geometry=gpd.points_from_xy(ggl_gdf.longitude, ggl_gdf.latitude),
    )
    ggl_gdf_gpd.set_crs(grid_land_overlap.crs, inplace=True)
    ggl_gdf_within = gpd.sjoin(
        ggl_gdf_gpd, grid_land_overlap, how="inner", predicate="within"
    )

    df_bld = ggl_gdf_within.groupby("id").size().reset_index(name="count")
    df_bld = df_bld.merge(grid_land_overlap, how="right")[
        ["id", "count"]
    ].fillna(0)
    df_bld = df_bld.rename(
        {"count": "total_buildings", "id": "grid_point_id"}, axis=1
    )

    # Add the directory containing IWI data to the system path
    sys.path.append(os.path.abspath("../src/datasources/05-IWI"))

    # Import the module for IWI data
    iwi = importlib.import_module("IWI_by_grid")

    df_iwi = iwi.get_IWI(ids_mun=grid_muni, shp=shp, save_to_blob=False)
    df_iwi = df_iwi[["grid_point_id", "IWI"]]

    # Merge features to create df_stationary
    # Also, add municipality info to the dataset
    grid_muni = grid_muni.rename({"id": "grid_point_id"}, axis=1).drop(
        "ADM2_PCODE", axis=1
    )
    df_stationary = df_iwi.merge(df_bld).merge(df_srtm).merge(grid_muni)

    return df_stationary


if __name__ == "__main__":
    # Pick Cell size
    cell_size = 0.5
    # Stationary features
    df_stationary = create_stationary_features(cell_size=cell_size)
    # Save to blob
    blob_name = (
        f"{PROJECT_PREFIX}/GRID_CELL_SIZE/{cell_size}/stationary_features.csv"
    )
    csv_data = df_stationary.to_csv(index=False)
    blob.upload_blob_data(blob_name=blob_name, data=csv_data)
