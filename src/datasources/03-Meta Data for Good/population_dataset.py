#!/usr/bin/env python3
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

# Directories
base_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features/"
)
input_dir = base_dir / "06_settlement/input/"
os.makedirs(input_dir, exist_ok=True)
grid_input_dir = base_dir / "02_housing_damage/output/"
output_dir = base_dir / "06_settlement/output/"
os.makedirs(output_dir, exist_ok=True)

# Load grid cells
grid = gpd.read_file(grid_input_dir / "hti_0.1_degree_grid_land_overlap.gpkg")


def get_population_data():
    # Open folder with data
    file_list = os.listdir(input_dir / "population_hti_2018-10-01")
    if ".DS_Store" in file_list:
        file_list.remove(".DS_Store")

    # Load raster
    pop_raster = rasterio.open(
        input_dir / "population_hti_2018-10-01" / file_list[0]
    )

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
    pop_out.to_csv(output_dir / "hti_population_data.csv", index=False)


if __name__ == "__main__":
    # Get population data
    get_population_data()
