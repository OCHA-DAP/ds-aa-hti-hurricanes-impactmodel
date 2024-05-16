#!/usr/bin/env python3
# From https://globaldatalab.org/areadata/table/iwi/

import ast
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Directories
base_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features/"
)
input_dir = base_dir / "05_vulnerability/input/"
os.makedirs(input_dir, exist_ok=True)
shp_input_dir = base_dir / "02_housing_damage/output/"
output_dir = base_dir / "05_vulnerability/output/"
os.makedirs(output_dir, exist_ok=True)

# Load ids of municipalities
ids_mun = pd.read_csv(
    base_dir / "02_housing_damage/input/grid_municipality_info.csv"
)
ids_mun.ADM1_EN = ids_mun.ADM1_EN.str.upper()

# Load grid cells
grid = gpd.read_file(
    base_dir / "02_housing_damage/output/hti_0.1_degree_grid_land_overlap.gpkg"
)

# Load IWI internationa;
iwi = pd.read_csv(input_dir / "GDL-Mean-International-Wealth-Index-(IWI).csv")

# Load Shapefile
shp = gpd.read_file(
    base_dir / "02_housing_damage/input/shapefile_hti_fixed.gpkg"
)
shp = shp.to_crs("EPSG:4326")


def get_IWI(country="Haiti"):
    iwi_country = iwi[(iwi.Country == country) & (iwi.Level == "Subnat")]
    # Manually fixing municipality names
    iwi_country = iwi_country[["Region", "2020"]].dropna()
    iwi_country["Region"] = iwi_country["Region"].str.upper()

    iwi_country["Region"] = iwi_country.Region.replace(
        "GRANDE-ANSE, NIPPES", "GRANDE'ANSE"
    )
    iwi_country["Region"] = iwi_country.Region.replace(
        "WEST (INCL METROPOLITAIN AREA)", "WEST"
    )
    iwi_country = pd.concat(
        [iwi_country, pd.DataFrame({"Region": ["NIPPES"], "2020": [28.7]})]
    )

    iwi_country = iwi_country.reset_index(drop=True)
    # Merge with Municipalities
    df_mun_iwi = iwi_country.merge(
        ids_mun, left_on="Region", right_on="ADM1_EN"
    ).drop_duplicates(subset=["id", "2020", "ADM1_EN"])

    # Merge with Shapefile
    IWI_shp = (
        df_mun_iwi.merge(shp, on="ADM1_PCODE", how="left")
        .drop_duplicates(subset=["id", "2020", "ADM1_PCODE"])[
            ["id", "2020", "geometry"]
        ]
        .reset_index(drop=True)
        .rename({"id": "grid_point_id", "2020": "IWI"}, axis=1)
    )

    # Save to csv
    IWI_shp[["grid_point_id", "IWI"]].to_csv(
        output_dir / "hti_iwi_bygrid.csv", index=False
    )


if __name__ == "__main__":
    # Get IWI subnational for Haiti
    get_IWI()
