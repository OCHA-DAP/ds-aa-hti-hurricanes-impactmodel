#!/usr/bin/env python3
# The data is from https://sites.research.google/open-buildings/#download
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Directories
base_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features/"
)
input_dir = base_dir / "02_housing_damage/input"
google_dir = input_dir / "Google Footprint Data/"

# Load tiles
google_tiles = gpd.read_file(google_dir / "tiles.geojson")

# Load  shp
shp = gpd.read_file(input_dir / "shapefile_hti_fixed.gpkg")
shp = shp.to_crs("EPSG:4326")

# Load grid cells
grid = gpd.read_file(
    base_dir / "02_housing_damage/output/hti_0.1_degree_grid_land_overlap.gpkg"
)

# Same CRS
shp.crs == google_tiles.crs
grid.crs == google_tiles.crs


def get_building_data():
    # Merge tiles and grid
    joined_df = gpd.sjoin(google_tiles, grid, how="right", op="intersects")
    file_pattern = joined_df.dropna().tile_id.unique()

    # Building data
    polygons_url_link = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/"
    points_url_link = "https://storage.googleapis.com/open-buildings-data/v3/points_s2_level_4_gzip/"
    file_list = [patt + "_buildings.csv.gz" for patt in file_pattern]

    # Downloading data. It takes a long time, be cautious.
    for file in file_list:
        r = requests.get(points_url_link + file, allow_redirects=True)
        open(google_dir / file, "wb").write(r.content)

    # Merging data (also takes time)
    google_df = pd.DataFrame()
    for file in file_list:
        zone_file = pd.read_csv(google_dir / file, compression="gzip")
        google_df = pd.concat([google_df, zone_file])

    # Creating geodataframe from df
    ggl_gdf = gpd.GeoDataFrame(
        google_df,
        geometry=gpd.points_from_xy(google_df.longitude, google_df.latitude),
    )
    ggl_gdf.set_crs(grid.crs, inplace=True)

    # SAVE DATA NOW (We'll delete the variable for resources purposes)
    google_df.to_csv(input_dir / "google_footprint_data.csv", index=False)

    # Join grid and building data
    del google_df
    ggl_gdf_within = gpd.sjoin(ggl_gdf, grid, how="inner", predicate="within")
    result = ggl_gdf_within.groupby("id").size().reset_index(name="count")
    result = result.merge(grid, how="right")[["id", "count"]].fillna(0)

    # SAVE BUILDINGS BY GRID AND BUILDING LOCATIONS
    result.to_csv(input_dir / "hti_google_bld_grid_count.csv", index=False)
    ggl_gdf_within.to_csv(
        input_dir / "hti_google_bld_points.csv.gz",
        compression="gzip",
        index=False,
    )


if __name__ == "__main__":
    # Building data at grid level
    get_building_data()
