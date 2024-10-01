#!/usr/bin/env python3
import io
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr

from src.utils import blob


def download_and_extract_smod(smod_link, temp_dir):
    """Download and extract the SMOD dataset to the given temporary directory."""
    req = requests.get(smod_link, verify=False, stream=True)
    with zipfile.ZipFile(io.BytesIO(req.content)) as zObj:
        fileNames = zObj.namelist()
        tif_file_name = None
        for fileName in fileNames:
            if fileName.endswith("tif"):
                tif_file_name = fileName
                # Write the file in the specified temp directory
                content = zObj.open(fileName).read()
                with open(temp_dir / fileName, "wb") as f:
                    f.write(content)
        if tif_file_name is not None:
            return tif_file_name
        else:
            raise ValueError("No TIF file found in the ZIP archive.")


def process_smod_raster(temp_dir, grid, tif_file_name):
    """Process the SMOD raster data and compute urban, rural, and water indices."""
    smod_raster = rxr.open_rasterio(temp_dir / tif_file_name)

    # Reproject to WGS84 and clip to grid bounds
    smod_raster_wgs84 = smod_raster.rio.reproject(grid.crs)
    smod_raster_wgs84_clip = smod_raster_wgs84.rio.clip_box(*grid.total_bounds)

    # Create a DataFrame for urban, rural, and water index
    smod_grid_vals = pd.DataFrame(
        {
            "id": grid["id"],
            "Centroid": grid["Centroid"],
            "urban": None,
            "rural": None,
            "water": None,
        }
    )

    # Calculate urban, rural, and water values for each grid cell
    for grd in grid.Centroid:
        grd_sel = grid[grid.Centroid == grd]
        grid_rast = smod_raster_wgs84_clip.rio.clip(
            grd_sel["geometry"], all_touched=False
        )
        smod_grid_vals.loc[grd_sel.index.values, ["urban"]] = (
            (grid_rast >= 21) & (grid_rast <= 30)
        ).sum().values / grid_rast.count().values
        smod_grid_vals.loc[grd_sel.index.values, ["rural"]] = (
            (grid_rast >= 11) & (grid_rast <= 13)
        ).sum().values / grid_rast.count().values
        smod_grid_vals.loc[grd_sel.index.values, ["water"]] = (
            grid_rast == 10
        ).sum().values / grid_rast.count().values

    return smod_grid_vals


def create_geodataframe(smod_grid_vals, grid):
    """Create a GeoDataFrame from the SMOD grid values and grid geometries."""
    return gpd.GeoDataFrame(
        smod_grid_vals.merge(grid, on="id")[
            ["id", "urban", "rural", "water", "geometry"]
        ],
        geometry="geometry",
    )


def main(smod_link, grid):
    """Main function to execute the workflow."""
    # Create a temporary directory that stays open
    temp_dir = Path(tempfile.mkdtemp())

    # Download and extract the SMOD dataset
    tif_file_name = download_and_extract_smod(smod_link, temp_dir)

    # Process the SMOD raster data and compute indices
    smod_grid_vals = process_smod_raster(temp_dir, grid, tif_file_name)

    # Create and return the GeoDataFrame
    df_urban_rural = create_geodataframe(smod_grid_vals, grid)

    # Clean up: manually remove the temporary directory and its contents
    for file in temp_dir.iterdir():
        file.unlink()
    temp_dir.rmdir()

    return df_urban_rural


if __name__ == "__main__":
    # Load data
    grid = blob.load_grid(complete=False)
    shp = blob.load_shp()

    # Data url
    smod_link = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2022A/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000/V1-0/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000_V1_0.zip"

    # Create dataset
    df_urban_rural = main(smod_link, grid)

    # Save to blob
    blob_name = "ds-aa-hti-hurricanes/urbanization/degree_of_urbanization.csv"
    data = df_urban_rural[["id", "urban", "rural", "water"]].to_csv(
        index=False
    )
    blob.upload_blob_data(blob_name=blob_name, data=data)
