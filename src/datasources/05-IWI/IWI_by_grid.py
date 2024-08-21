#!/usr/bin/env python3
# From https://globaldatalab.org/areadata/table/iwi/

import ast
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

def get_IWI(ids_mun, shp, save_to_blob=True, country="Haiti"):
    # Load IWI international
    iwi_dir = (
        PROJECT_PREFIX
        + "/vulnerability/input_dir/GDL-Mean-International-Wealth-Index-(IWI).csv"
    )
    iwi = blob.load_csv(iwi_dir)

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
    ids_mun.ADM1_EN = ids_mun.ADM1_EN.str.upper()
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
    if save_to_blob:
        # Save to blob
        csv_data = IWI_shp[["grid_point_id", "IWI"]].to_csv(index=False)
        blob_path = PROJECT_PREFIX + "/vulnerability/output_dir/hti_iwi_bygrid.csv"
        blob.upload_blob_data(blob_path, csv_data)
    else:
        return IWI_shp


if __name__ == "__main__":
    # Load ids of municipalities
    ids_mun = blob.load_csv(
        PROJECT_PREFIX + "/grid/input_dir/grid_municipality_info.csv"
    )

    # Load Shapefile
    shp = blob.load_shp()
    shp = shp.to_crs("EPSG:4326")
    # Get IWI subnational for Haiti
    get_IWI(
        ids_mun=ids_mun, 
        shp=shp
        )
