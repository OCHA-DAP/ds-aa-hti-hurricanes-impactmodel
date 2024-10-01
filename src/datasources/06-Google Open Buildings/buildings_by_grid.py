#!/usr/bin/env python3
# The data is from https://sites.research.google/open-buildings/#download
import os
import tempfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

# Load tiles
tiles_path = PROJECT_PREFIX + "/google/input_dir/tiles.geojson"
google_tiles = blob.load_gdf_from_blob(tiles_path)

# Load Shapefile
shp = blob.load_shp()
shp = shp.to_crs("EPSG:4326")

# Load grid cells
grid = blob.load_grid(complete=False)

# Same CRS
shp.crs == google_tiles.crs
grid.crs == google_tiles.crs


def get_building_data(
    PROJECT_PREFIX=PROJECT_PREFIX, google_tiles=google_tiles, grid=grid
):
    # Merge tiles and grid
    joined_df = gpd.sjoin(
        google_tiles, grid, how="right", predicate="intersects"
    )
    file_pattern = joined_df.dropna().tile_id.unique()

    # Building data URLs
    polygons_url_link = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/"
    points_url_link = "https://storage.googleapis.com/open-buildings-data/v3/points_s2_level_4_gzip/"
    file_list = [patt + "_buildings.csv.gz" for patt in file_pattern]

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        google_dir = Path(tmp_dir)

        # Downloading data. It takes a long time, be cautious.
        for file in file_list:
            r = requests.get(points_url_link + file, allow_redirects=True)
            with open(google_dir / file, "wb") as f:
                f.write(r.content)

        # Merging data (also takes time)
        google_df = pd.DataFrame()
        for file in file_list:
            zone_file = pd.read_csv(google_dir / file, compression="gzip")
            google_df = pd.concat([google_df, zone_file])

        # Creating geodataframe from df
        ggl_gdf = gpd.GeoDataFrame(
            google_df,
            geometry=gpd.points_from_xy(
                google_df.longitude, google_df.latitude
            ),
        )
        ggl_gdf.set_crs(grid.crs, inplace=True)

        # SAVE DATA NOW (We'll delete the variable for resources purposes)
        csv_data = google_df.to_csv(index=False)
        data_path = (
            f"{PROJECT_PREFIX}/google/input_dir/google_footprint_data.csv"
        )
        blob.upload_blob_data(data_path, csv_data)

        # Join grid and building data
        del google_df
        del csv_data
        ggl_gdf_within = gpd.sjoin(
            ggl_gdf, grid, how="inner", predicate="within"
        )
        result = ggl_gdf_within.groupby("id").size().reset_index(name="count")
        result = result.merge(grid, how="right")[["id", "count"]].fillna(0)

        # SAVE BUILDINGS BY GRID AND BUILDING LOCATIONS
        csv_results = result.to_csv(index=False)
        results_path = (
            f"{PROJECT_PREFIX}/google/output_dir/hti_google_bld_grid_count.csv"
        )
        blob.upload_blob_data(results_path, csv_results)

        points_csv = ggl_gdf_within.to_csv(index=False)
        points_path = (
            f"{PROJECT_PREFIX}/google/output_dir/hti_google_bld_points.csv"
        )
        blob.upload_blob_data(points_path, points_csv)


if __name__ == "__main__":
    # Building data at grid level
    get_building_data()
