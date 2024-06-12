#!/usr/bin/env python3
import os
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

# Load grid cells
grid = blob.load_grid(complete=False)


def get_population_data(PROJECT_PREFIX=PROJECT_PREFIX, prod_dev="dev"):
    # Define paths
    input_blob_path = (
        f"{PROJECT_PREFIX}/settlement/input_dir/population_hti_2018-10-01.tif"
    )
    output_blob_path = (
        f"{PROJECT_PREFIX}/settlement/output_dir/hti_population_data.csv"
    )

    # List and load the TIFF file from blob storage
    tif_data = blob.load_blob_data(input_blob_path, prod_dev=prod_dev)
    pop_raster = rasterio.open(BytesIO(tif_data))

    # Get pop data
    pop = pop_raster.read(1)
    summary_stats = zonal_stats(
        grid,
        pop,
        stats=["sum"],
        nodata=-9999,
        all_touched=True,
        affine=pop_raster.transform,
    )
    grid_pop = pd.DataFrame(summary_stats)

    # Merge with grid to get the geometry
    grid_pop_df = pd.concat([grid, grid_pop], axis=1)
    grid_pop_df = grid_pop_df.fillna(0)
    grid_pop_df = grid_pop_df.rename({"sum": "total_pop"}, axis=1)

    # Save population data by grid to csv
    pop_out = grid_pop_df[["id", "total_pop"]]
    csv_data = pop_out.to_csv(index=False)

    # Upload the CSV data to blob storage
    blob.upload_blob_data(
        blob_name=output_blob_path, data=csv_data, prod_dev=prod_dev
    )


if __name__ == "__main__":
    # Get population data
    get_population_data()
