#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.merge import merge
from rasterstats import zonal_stats
from shapely.geometry import Point

""" https://dwtkns.com/srtm30m/ to get the data (must be registered)"""

# Directories
data_dir = os.getenv("STORM_DATA_DIR")
base_url = Path(data_dir) / "analysis_hti/02_model_features/"
input_dir = base_url / "04_topography/input/srtm/"
os.makedirs(input_dir, exist_ok=True)
output_dir = base_url / "04_topography/output/"
os.makedirs(output_dir, exist_ok=True)
shp_output_dir = base_url / "02_housing_damage/output/"

# Load grid
grid = gpd.read_file(
    base_url / "02_housing_damage/output/hti_0.1_degree_grid_land_overlap.gpkg"
)

# Load shapefile
shp = gpd.read_file(
    base_url / "02_housing_damage/input/shapefile_hti_fixed.gpkg"
)
shp = shp.to_crs("EPSG:4326")


def merge_raster_tiles():
    # List all files
    fileList = os.listdir(input_dir / "dwtkns")

    # All .hgt together
    mosaic_raster = []
    for file in fileList:
        if file.endswith(".hgt.zip"):
            rast = rasterio.open(input_dir / "dwtkns" / file)
            mosaic_raster.append(rast)
    merged_raster, out_raster = merge(mosaic_raster)

    # Metadata of the files
    out_meta = rast.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": merged_raster.shape[1],
            "width": merged_raster.shape[2],
            "transform": out_raster,
        }
    )

    # Save file
    with rasterio.open(
        input_dir / "hti_merged_srtm.tif", "w", **out_meta
    ) as dest:
        dest.write(merged_raster)


def get_topography_features():
    # Load merged raster
    merged_rast = rasterio.open(input_dir / "hti_merged_srtm.tif")
    merged_rast_path = input_dir / "hti_merged_srtm.tif"
    ## Altitude
    altitude = merged_rast.read(1)

    ## MEAN ALTITUDE by grid
    summary_stats = zonal_stats(
        grid,
        altitude,
        stats=["mean"],
        nodata=-32768,
        all_touched=True,
        affine=merged_rast.transform,
    )

    grid_elev = pd.DataFrame(summary_stats)
    grid_elev_df = pd.concat([grid, grid_elev], axis=1)
    del altitude

    ## slope
    hti_slope_gdaldem = input_dir / "hti_slope_gdaldem.tif"

    # !gdaldem slope -s 111120 -co COMPRESS=DEFLATE -co ZLEVEL=9 \
    # "{merged_rast_path}" "{hti_slope_gdaldem}" -compute_edges
    subprocess.run(
        [
            "gdaldem",
            "slope",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "ZLEVEL=9",
            merged_rast_path,
            hti_slope_gdaldem,
            "-compute_edges",
        ],
        check=True,
    )

    slope_rast = rasterio.open(input_dir / "hti_slope_gdaldem.tif")
    slope_array = slope_rast.read(1)

    ## MEAN SLOPE by grid
    summary_stats = zonal_stats(
        grid,
        slope_array,
        stats=["mean", "std"],
        nodata=-9999,
        all_touched=True,
        affine=merged_rast.transform,
    )

    grid_slope = pd.DataFrame(summary_stats)
    grid_slope_df = pd.concat([grid, grid_slope], axis=1)
    del slope_array

    ## calculate  Terrain Ruggedness Index TRI
    hti_tri_gdaldem = input_dir / "hti_tri_gdaldem.tif"
    # !gdaldem TRI -co COMPRESS=DEFLATE -co ZLEVEL=9 \
    # "{merged_rast_path}" "{hti_tri_gdaldem}" -compute_edges
    subprocess.run(
        [
            "gdaldem",
            "TRI",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "ZLEVEL=9",
            merged_rast_path,
            hti_tri_gdaldem,
            "-compute_edges",
        ],
        check=True,
    )
    tri_rast = rasterio.open(input_dir / "hti_tri_gdaldem.tif")
    tri_array = tri_rast.read(1)

    ## MEAN RUGGEDNESS by grid
    summary_stats = zonal_stats(
        grid,
        tri_array,
        stats=["mean", "std"],
        nodata=-9999,
        all_touched=True,
        affine=merged_rast.transform,
    )

    grid_rudg = pd.DataFrame(summary_stats)
    grid_rudg_df = pd.concat([grid, grid_rudg], axis=1)
    del tri_array

    # Dataframes

    df_slope = grid_slope_df.fillna(0).rename({"mean": "mean_slope"}, axis=1)
    df_elev = grid_elev_df.fillna(0).rename({"mean": "mean_elev"}, axis=1)
    df_rug = grid_rudg_df.fillna(0).rename({"mean": "mean_rug"}, axis=1)
    df_slope_elev = df_slope.merge(df_elev, on="id")
    df_terrain = df_slope_elev.merge(df_rug, on="id")[
        ["id", "mean_elev", "mean_slope", "mean_rug"]
    ]

    return df_terrain


def get_coast_features():
    # dissolving polygons into one land mass
    dissolved_shp = shp.dissolve(by="ADM0_PCODE")

    # Coastline
    coastline = dissolved_shp.boundary
    coastline.crs = grid.crs

    # Linestrings and MultiLinestrings
    grid_line_gdf = gpd.overlay(
        gpd.GeoDataFrame(
            coastline, geometry=coastline.geometry, crs=coastline.crs
        ).reset_index(),
        grid,
        how="intersection",
    )[["id", "Centroid", "geometry"]]
    # Coast line
    grid_line_gdf["coast_length"] = (
        grid_line_gdf["geometry"].to_crs(25394).length
    )  # 25394 gives me length in meters.
    grid_coast = grid[["id", "Centroid"]].merge(
        grid_line_gdf, on=["id", "Centroid"], how="left"
    )

    # With coast? Binary
    grid_coast["with_coast"] = np.where(grid_coast["coast_length"] > 0, 1, 0)
    grid_coast["with_coast"].value_counts()

    return grid_coast


if __name__ == "__main__":
    # Merge raster tiles
    merge_raster_tiles()
    # Topograpgy features (ELEV,SLOPE,RUGG)
    df_terrain = get_topography_features()
    # Coast related features
    grid_coast = get_coast_features()

    # Merge data
    merge_final = df_terrain.merge(grid_coast, on="id", how="left")
    merge_final = merge_final[
        [
            "id",
            "with_coast",
            "coast_length",
            "mean_elev",
            "mean_slope",
            "mean_rug",
        ]
    ]
    merge_final = merge_final.fillna(0)  # No coast length? Then is 0

    merge_final.to_csv(
        output_dir / "topography_variables_bygrid.csv", index=False
    )
