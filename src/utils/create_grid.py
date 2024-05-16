#!/usr/bin/env python3
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# %% Input and output dir
input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/input/"
)
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/output/"
)

# Load shapefile
shp = gpd.read_file(input_dir / "shapefile_hti_fixed.gpkg")
shp = shp.to_crs("EPSG:4326")

# %% Define grid
xmin, xmax, ymin, ymax = -75, -71, 17, 21  # Vietnam extremes coordintates

cell_size = 0.1

cols = list(np.arange(xmin, xmax + cell_size, cell_size))
rows = list(np.arange(ymin, ymax + cell_size, cell_size))
rows.reverse()

polygons = [
    Polygon(
        [
            (x, y),
            (x + cell_size, y),
            (x + cell_size, y - cell_size),
            (x, y - cell_size),
        ]
    )
    for x in cols
    for y in rows
]
grid = gpd.GeoDataFrame({"geometry": polygons}, crs=shp.crs)
grid["id"] = grid.index + 1


# %% Centroids
# Extract lat and lon from the centerpoint
grid["Longitude"] = grid["geometry"].centroid.map(lambda p: p.x)
grid["Latitude"] = grid["geometry"].centroid.map(lambda p: p.y)
grid["Centroid"] = (
    round(grid["Longitude"], 2).astype(str)
    + "W"
    + "_"
    + round(grid["Latitude"], 2).astype(str)
    + "N"
)

grid_centroids = grid.copy()
grid_centroids["geometry"] = grid_centroids["geometry"].centroid
grid_centroids.loc[:, "geometry"].plot()

# %% intersection of grid and shapefile
adm2_grid_intersection = gpd.overlay(shp, grid, how="identity")
adm2_grid_intersection = adm2_grid_intersection.dropna(subset=["id"])
grid_land_overlap = grid.loc[grid["id"].isin(adm2_grid_intersection["id"])]

# Centroids of intersection
grid_land_overlap_centroids = grid_centroids.loc[
    grid["id"].isin(adm2_grid_intersection["id"])
]

# %% Grids by municipality
grid_muni = gpd.sjoin(shp, grid_land_overlap, how="inner")

intersection_areas = []
for index, row in grid_muni.iterrows():
    id_cell = row["id"]
    grid_cell = grid_land_overlap[grid_land_overlap.id == id_cell].geometry
    municipality_polygon = row["geometry"]
    intersection_area = grid_cell.intersection(municipality_polygon).area
    intersection_areas.append(intersection_area)

# Add area of intersection to each row
grid_muni["intersection_area"] = [x.array[0] for x in intersection_areas]

# Find the municipality with the largest intersection area for each grid centroid and drop the rest
grid_muni = grid_muni.sort_values("intersection_area", ascending=False)
grid_muni_total = grid_muni.drop_duplicates(subset="id", keep="first")

admin2_grid_muni = (
    grid_muni_total[["id", "ADM2_EN"]]
    .groupby(by="ADM2_EN")
    .count()
    .rename({"id": "grid_cells"}, axis=1)
    .sort_values("grid_cells", ascending=False)
)
admin1_grid_muni = (
    grid_muni_total[["id", "ADM1_FR"]]
    .groupby(by="ADM1_FR")
    .count()
    .rename({"id": "grid_cells"}, axis=1)
    .sort_values("grid_cells", ascending=False)
)


# %% SAVE EVERYTHING
grid.to_file(output_dir / "hti_0.1_degree_grid.gpkg", driver="GPKG")
grid_centroids.to_file(
    output_dir / "hti_0.1_degree_grid_centroids.gpkg", driver="GPKG"
)
adm2_grid_intersection.to_file(
    input_dir / "hti_adm2_grid_intersection.gpkg", driver="GPKG"
)
grid_land_overlap.to_file(
    output_dir / "hti_0.1_degree_grid_land_overlap.gpkg", driver="GPKG"
)
grid_land_overlap_centroids.to_file(
    output_dir / "hti_0.1_degree_grid_centroids_land_overlap.gpkg",
    driver="GPKG",
)
grid_muni_total[
    ["id", "ADM1_FR", "ADM1_PCODE", "ADM1_EN", "ADM2_EN", "ADM2_PCODE"]
].to_csv(input_dir / "grid_municipality_info.csv", index=False)
