#!/usr/bin/env python3
import os
import subprocess
import tempfile
import zipfile
from io import BytesIO
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

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

""" https://dwtkns.com/srtm30m/ to get the data (must be registered)"""

# Load grid cells
grid = blob.load_grid(complete=False)

# Load shapefile
shp = blob.load_shp()
shp = shp.to_crs("EPSG:4326")


def load_hgt_zip_from_blob(blob_path, prod_dev="dev"):
    # Load the .hgt.zip file data from blob storage
    hgt_zip_data = blob.load_blob_data(blob_path, prod_dev=prod_dev)

    # Extract the contents of the .hgt.zip file to a temporary folder (/tmp)
    with zipfile.ZipFile(BytesIO(hgt_zip_data), "r") as zip_ref:
        # Extract all contents to a temporary directory (/tmp)
        zip_ref.extractall("/tmp")

    # Return the path of the temporary directory
    return "/tmp"


def merge_raster_tiles(PROJECT_PREFIX=PROJECT_PREFIX, prod_dev="dev"):
    # Define paths
    input_blob_path = f"{PROJECT_PREFIX}/topography/input_dir/dwtkns"
    output_blob_path = (
        f"{PROJECT_PREFIX}/topography/input_dir/hti_merged_srtm.tif"
    )

    # List all files in the input directory
    blob_files = blob.list_container_blobs(input_blob_path, prod_dev=prod_dev)

    # Load .hgt.zip files, extract .hgt files, and merge
    mosaic_raster = []
    for blob_file in blob_files:
        if blob_file.endswith(".hgt.zip"):
            # Load .hgt.zip file from blob storage and extract .hgt files to a temporary folder
            tmp_folder = load_hgt_zip_from_blob(blob_file, prod_dev=prod_dev)
            # List all files in the temporary folder
            hgt_files = [
                f for f in os.listdir(tmp_folder) if f.endswith(".hgt")
            ]
            # Load each .hgt file, convert it to a rasterio object, and append it to mosaic_raster
            for hgt_file in hgt_files:
                # Open each .hgt file and append the dataset to mosaic_raster_datasets
                for hgt_file in hgt_files:
                    rast = rasterio.open(os.path.join(tmp_folder, hgt_file))
                    mosaic_raster.append(rast)

    # Merge raster tiles
    if mosaic_raster:
        merged_raster_data, out_raster = merge(mosaic_raster)
        # Metadata of the files
        out_meta = rast.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": merged_raster_data.shape[1],
                "width": merged_raster_data.shape[2],
                "transform": out_raster,
            }
        )

        # Write merged raster to blob storage
        with BytesIO() as dest:
            with rasterio.open(dest, "w", **out_meta) as dest_raster:
                dest_raster.write(merged_raster_data)
            dest.seek(0)
            merged_data = dest.read()
            blob.upload_blob_data(
                output_blob_path, merged_data, prod_dev=prod_dev
            )


def get_topography_features(PROJECT_PREFIX=PROJECT_PREFIX, prod_dev="dev"):
    # Define paths
    input_blob_path = (
        f"{PROJECT_PREFIX}/topography/input_dir/hti_merged_srtm.tif"
    )

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download the merged raster locally
        merged_rast_data = blob.load_blob_data(
            input_blob_path, prod_dev=prod_dev
        )
        merged_rast_path = os.path.join(tmp_dir, "hti_merged_srtm.tif")
        with open(merged_rast_path, "wb") as f:
            f.write(merged_rast_data)

        # Load merged raster
        merged_rast = rasterio.open(merged_rast_path)

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

        ## Slope
        hti_slope_gdaldem = os.path.join(tmp_dir, "hti_slope_gdaldem.tif")

        # Run gdaldem slope command
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

        slope_rast = rasterio.open(hti_slope_gdaldem)
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

        ## Calculate Terrain Ruggedness Index (TRI)
        hti_tri_gdaldem = os.path.join(tmp_dir, "hti_tri_gdaldem.tif")

        # Run gdaldem TRI command
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

        tri_rast = rasterio.open(hti_tri_gdaldem)
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

        grid_ruggedness = pd.DataFrame(summary_stats)
        grid_ruggedness_df = pd.concat([grid, grid_ruggedness], axis=1)
        del tri_array

        # Dataframes
        df_slope = grid_slope_df.fillna(0).rename(
            {"mean": "mean_slope"}, axis=1
        )
        df_elev = grid_elev_df.fillna(0).rename({"mean": "mean_elev"}, axis=1)
        df_rugged = grid_ruggedness_df.fillna(0).rename(
            {"mean": "mean_rug"}, axis=1
        )

        df_slope_elev = df_slope.merge(df_elev, on="id")
        df_terrain = df_slope_elev.merge(df_rugged, on="id")[
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
    # merge_raster_tiles()
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

    # Save to blob
    csv_data = merge_final.to_csv(index=False)
    blob_path = (
        PROJECT_PREFIX
        + "/topography/output_dir/topography_variables_bygrid.csv"
    )
    blob.upload_blob_data(blob_path, csv_data)
