#!/usr/bin/env python3
import datetime as dt
import getpass
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from bs4 import BeautifulSoup
from rasterstats import zonal_stats

# To create an account for downloading the data
# follow the instructions here: https://registration.pps.eosdis.nasa.gov/registration/
# Change the user name and provide the password in the code
# Normally, is just the passwords and the user is just the email that you registered.
# Normally, is just the passwords and the user is just the email that you registered.
USERNAME = getpass.getpass(prompt="Username: ", stream=None)
PASSWORD = getpass.getpass(prompt="Password: ", stream=None)


def load_metadata():
    # Setting directories
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/input"
    )
    # Load and clean the typhoon metadata
    # We really only care about the landfall date
    typhoon_metadata = pd.read_csv(
        input_dir / "metadata_typhoons.csv"
    ).set_index("typhoon")
    for colname in ["startdate", "enddate", "landfalldate"]:
        typhoon_metadata[colname] = pd.to_datetime(
            typhoon_metadata[colname], format="%Y-%m-%d"
        )
    return typhoon_metadata


def load_grid():
    grid_input = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/02_housing_damage/output/"
    )

    # Load grid
    grid_land_overlap = gpd.read_file(
        grid_input / "hti_0.1_degree_grid_land_overlap.gpkg"
    )
    grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
    grid = grid_land_overlap.copy()
    return grid


def list_files(url, USERNAME=USERNAME, PASSWORD=PASSWORD):
    page = requests.get(url, auth=(USERNAME, PASSWORD)).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith("tif")
    ]


def download_gpm_http(
    start_date, end_date, download_path, USERNAME=USERNAME, PASSWORD=PASSWORD
):
    base_url = "https://arthurhouhttps.pps.eosdis.nasa.gov/pub/gpmdata"

    date_list = pd.date_range(start_date, end_date)
    file_list = []

    for date in date_list:
        # print(f"Downloading data for date {date}")
        day_path = download_path / date.strftime("%Y%m%d")
        day_path.mkdir(parents=True, exist_ok=True)

        url = f"{base_url}/{date.strftime('%Y/%m/%d')}/gis"
        tiff_files = list_files(url=url, USERNAME=USERNAME, PASSWORD=PASSWORD)

        for tiff_file in tiff_files:
            file_name = tiff_file.split("/")[-1]

            file_path = day_path / file_name
            file_list.append(file_path)
            r = requests.get(tiff_file, auth=(USERNAME, PASSWORD))
            time.sleep(0.2)
            open(file_path, "wb").write(r.content)

    return file_list


# Download data
def download_rainfall_data(
    USERNAME=USERNAME, PASSWORD=PASSWORD, DAYS_TO_LANDFALL=2
):
    # Setting directories
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/input"
    )
    # Setting path to save the GPM data
    gpm_file_name = "gpm_data/rainfall_data/output_hhr/"
    gpm_folder_path = Path(input_dir, gpm_file_name)
    typhoon_metadata = load_metadata()

    i = 0
    for typhoon, metadata in typhoon_metadata[i:].iterrows():
        start_date = metadata["landfalldate"] - dt.timedelta(
            days=DAYS_TO_LANDFALL
        )
        end_date = metadata["landfalldate"] + dt.timedelta(
            days=DAYS_TO_LANDFALL
        )
        print("Typhoon {}/{}".format(i, len(typhoon_metadata) - 1))
        i += 1
        print(
            f"Downloading data for {typhoon} between {start_date} and {end_date}"
        )
        download_gpm_http(
            start_date=start_date,
            end_date=end_date,
            download_path=gpm_folder_path / typhoon / "GPM",
            USERNAME=USERNAME,
        )


def create_rainfall_dataset():
    # setting up loop for running through all typhoons
    # extracting the max and mean of the 4 adjacent cells due to shifting to grids

    # Setting directories
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/input"
    )

    # Output dir
    processed_output_dir = Path(
        input_dir / "gpm_data/rainfall_data/output_hhr_processed/"
    )
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    # Load grid
    grid = load_grid()

    # Load gpm data
    gpm_file_name = "gpm_data/rainfall_data/output_hhr/"
    gpm_folder_path = Path(input_dir, gpm_file_name)
    typhoon_list = os.listdir(gpm_folder_path)
    stats_list = ["mean", "max"]
    j = 0
    for typ in typhoon_list[j:]:
        print(typ, j)
        j += 1
        day_list = os.listdir(gpm_folder_path / typ / "GPM")
        if ".DS_Store" in day_list:  # Sometimes this causes problems
            day_list.remove(".DS_Store")
        day_df = pd.DataFrame()
        for day in day_list:
            file_list = os.listdir(gpm_folder_path / typ / "GPM" / day)
            file_df = pd.DataFrame()
            for file in file_list:
                if file.startswith("3B-HHR"):
                    file_path = Path(
                        gpm_folder_path / typ / "GPM" / day / file
                    )
                    input_raster = rasterio.open(file_path)
                    array = input_raster.read(1)
                    summary_stats = zonal_stats(
                        grid,
                        array,
                        stats=stats_list,
                        nodata=29999,
                        all_touched=True,
                        affine=input_raster.transform,
                    )
                    grid_stats = pd.DataFrame(summary_stats)
                    # change values by dividing by 10 to mm/hr
                    grid_stats[stats_list] /= 10
                    grid_merged = pd.merge(
                        grid.drop(
                            ["geometry", "Longitude", "Latitude"], axis=1
                        ),
                        grid_stats,
                        left_index=True,
                        right_index=True,
                    )
                    grid_merged["start"] = "%s%s:%s%s:%s%s" % (
                        *file.split("-S")[1][0:6],
                    )
                    grid_merged["end"] = "%s%s:%s%s:%s%s" % (
                        *file.split("-E")[1][0:6],
                    )

                    file_df = pd.concat([file_df, grid_merged], axis=0)
            file_df["date"] = str(day)
            day_df = pd.concat([day_df, file_df], axis=0)
        day_df["time"] = day_df["date"].astype(str) + "_" + day_df["start"]
        for stats in stats_list:
            day_wide = pd.pivot(
                day_df,
                index=["id", "Centroid"],
                columns=["time"],
                values=[stats],
            )
            day_wide.columns = day_wide.columns.droplevel(0)
            day_wide.reset_index(inplace=True)
            day_wide.to_csv(
                processed_output_dir
                / str(typ + "_gridstats_" + stats + ".csv"),
                index=False,
            )


def compute_stats():
    # Load directories
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/input"
    )
    processed_output_dir = (
        input_dir / "gpm_data/rainfall_data/output_hhr_processed/"
    )
    output_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_hti/02_model_features/03_rainfall/output"
    )
    output_dir.mkdir(exist_ok=True)

    # Load metadata
    typhoon_metadata = load_metadata()
    # To make sure the dates can be converted to datetype
    typhoon_metadata["startdate"] = [
        str_col.replace("/", "-") for str_col in typhoon_metadata["startdate"]
    ]
    typhoon_metadata["enddate"] = [
        str_col.replace("/", "-") for str_col in typhoon_metadata["enddate"]
    ]
    typhoon_metadata["landfalldate"] = [
        str_col.replace("/", "-")
        for str_col in typhoon_metadata["landfalldate"]
    ]

    typhoon_metadata["landfall_date_time"] = (
        typhoon_metadata["landfalldate"]
        + "-"
        + typhoon_metadata["landfall_time"]
    )

    typhoons = list(typhoon_metadata["typhoon"].values)
    # Defining windows
    time_frame_24 = 48  # in half hours
    time_frame_6 = 12  # in half hours
    mov_window = 12  # in half hours
    before_landfall_h = 72  # how many hours before landfall to include
    after_landfall_h = 72  # how many hours before landfall to include

    # looping over all typhoons
    for stats in ["mean", "max"]:
        df_rainfall_final = pd.DataFrame(
            columns=["typhoon", "id", "Centroid", "rainfall_Total"]
        )
        for typ in typhoons:
            print(typ)
            # Getting typhoon info
            df_info = typhoon_metadata[typhoon_metadata["typhoon"] == typ]
            landfall = df_info["landfall_date_time"].values[0]
            landfall = dt.datetime.strptime(landfall, "%Y-%m-%d-%H:%M:%S")
            # End date is landfall date
            # Start date is 72 hours before landfall date
            # end_date = landfall
            end_date = landfall + dt.timedelta(
                hours=after_landfall_h
            )  # landfall
            # start_date = end_date - datetime.timedelta(hours=before_landfall_h)
            start_date = landfall - dt.timedelta(hours=before_landfall_h)
            # Loading the data
            df_rainfall = pd.read_csv(
                processed_output_dir
                / str(typ + "_gridstats_" + stats + ".csv")
            )
            # Convert column names to date format
            for col in df_rainfall.columns[2:]:
                date_format = dt.datetime.strptime(col, "%Y%m%d_%H:%M:%S")
                df_rainfall = df_rainfall.rename(columns={col: date_format})

            df_mean_rainfall = pd.DataFrame(
                {"id": df_rainfall["id"], "Centroid": df_rainfall["Centroid"]}
            )
            available_dates_t = [
                date
                for date in df_rainfall.columns[2:]
                if (date >= start_date) & (date < end_date)
            ]
            #####################################
            df_mean_rainfall["rainfall_max_6h"] = (
                df_rainfall.iloc[:, 2:]
                .rolling(time_frame_6, axis=1)
                .mean()
                .max(axis=1)
            )

            df_mean_rainfall["rainfall_max_24h"] = (
                df_rainfall.iloc[:, 2:]
                .rolling(time_frame_24, axis=1)
                .mean()
                .max(axis=1)
            )

            df_mean_rainfall["rainfall_Total"] = 0.5 * df_rainfall[
                available_dates_t
            ].sum(axis=1)

            df_rainfall_single = df_mean_rainfall[
                [
                    "id",
                    "Centroid",
                    "rainfall_max_6h",
                    "rainfall_max_24h",
                    "rainfall_Total",
                ]
            ]
            df_rainfall_single["typhoon"] = typ
            df_rainfall_final = pd.concat(
                [df_rainfall_final, df_rainfall_single]
            )
        df_rainfall_final.to_csv(
            output_dir / str("rainfall_data_rw_" + stats + ".csv"), index=False
        )


if __name__ == "__main__":
    # Download data
    download_rainfall_data()
    # Create datasets
    create_rainfall_dataset()
    # Stats at grid level
    compute_stats()
