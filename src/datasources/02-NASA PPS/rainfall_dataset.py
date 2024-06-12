#!/usr/bin/env python3
import datetime as dt
import getpass
import os
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from bs4 import BeautifulSoup
from rasterstats import zonal_stats

from src.utils import blob

# To create an account for downloading the data
# follow the instructions here: https://registration.pps.eosdis.nasa.gov/registration/
# Change the user name and provide the password in the code
# Normally, is just the passwords and the user is just the email that you registered.
# Normally, is just the passwords and the user is just the email that you registered.
USERNAME = getpass.getpass(prompt="Username: ", stream=None)
PASSWORD = getpass.getpass(prompt="Password: ", stream=None)
PROJECT_PREFIX = "ds-aa-hti-hurricanes"


def load_metadata():
    # Load and clean the typhoon metadata
    # We really only care about the landfall date
    typhoon_metadata = blob.load_metadata().set_index("typhoon")

    for colname in ["startdate", "enddate", "landfalldate"]:
        typhoon_metadata[colname] = pd.to_datetime(
            typhoon_metadata[colname], format="%Y-%m-%d"
        )
    return typhoon_metadata


def load_grid():
    # Load grid
    grid_land_overlap = blob.load_grid(complete=False)
    grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
    return grid_land_overlap


def list_files(url, USERNAME=USERNAME, PASSWORD=PASSWORD):
    page = requests.get(url, auth=(USERNAME, PASSWORD)).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith("tif")
    ]


# Function to download and upload GPM data to blob storage
def download_gpm_http(start_date, end_date, USERNAME, PASSWORD):
    base_url = "https://arthurhouhttps.pps.eosdis.nasa.gov/pub/gpmdata"
    date_list = pd.date_range(start_date, end_date)
    file_list = []

    for date in date_list:
        url = f"{base_url}/{date.strftime('%Y/%m/%d')}/gis"
        tiff_files = list_files(url=url, USERNAME=USERNAME, PASSWORD=PASSWORD)

        for tiff_file in tiff_files:
            file_name = tiff_file.split("/")[-1]
            blob_path = f"{PROJECT_PREFIX}/rainfall/input_dir/gpm_data/rainfall_data/output_hhr/{date.strftime('%Y%m%d')}/{file_name}"
            r = requests.get(tiff_file, auth=(USERNAME, PASSWORD))
            time.sleep(0.2)
            blob.upload_blob_data(
                blob_name=blob_path, data=r.content, prod_dev="dev"
            )
            file_list.append(blob_path)

    return file_list


# Function to download rainfall data and upload to blob storage
def download_rainfall_data(USERNAME, PASSWORD, DAYS_TO_LANDFALL=2):
    typhoon_metadata = load_metadata()
    i = 0

    for typhoon, metadata in typhoon_metadata.iterrows():
        start_date = metadata["landfalldate"] - dt.timedelta(
            days=DAYS_TO_LANDFALL
        )
        end_date = metadata["landfalldate"] + dt.timedelta(
            days=DAYS_TO_LANDFALL
        )
        print(f"Typhoon {i}/{len(typhoon_metadata)-1}")
        print(
            f"Downloading data for {typhoon} between {start_date} and {end_date}"
        )
        i += 1
        download_gpm_http(
            start_date=start_date,
            end_date=end_date,
            USERNAME=USERNAME,
            PASSWORD=PASSWORD,
        )


def create_rainfall_dataset(prod_dev="dev"):
    # Load grid
    grid = load_grid()

    # Path to GPM data
    gpm_folder_path = (
        f"{PROJECT_PREFIX}/rainfall/gpm_data/rainfall_data/output_hhr/"
    )
    typhoon_list = [
        blob.split("/")[-2]
        for blob in blob.list_container_blobs(
            name_starts_with=gpm_folder_path, prod_dev=prod_dev
        )
        if blob.endswith("/")
    ]

    stats_list = ["mean", "max"]
    j = 0

    for typ in typhoon_list[j:]:
        print(typ, j)
        j += 1

        # Get the list of days for this typhoon from the blob storage
        typhoon_path = f"{gpm_folder_path}{typ}/GPM/"
        day_list = [
            blob.split("/")[-2]
            for blob in blob.list_container_blobs(
                name_starts_with=typhoon_path, prod_dev=prod_dev
            )
            if blob.endswith("/")
        ]

        day_df = pd.DataFrame()

        for day in day_list:
            day_path = f"{typhoon_path}{day}/"
            file_list = [
                blob.split("/")[-1]
                for blob in blob.list_container_blobs(
                    name_starts_with=day_path, prod_dev=prod_dev
                )
                if blob.startswith("3B-HHR")
            ]

            file_df = pd.DataFrame()

            for file in file_list:
                file_path = f"{day_path}{file}"
                input_raster = rasterio.open(
                    BytesIO(blob.load_blob_data(file_path, prod_dev=prod_dev))
                )
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
                    grid.drop(["geometry", "Longitude", "Latitude"], axis=1),
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

            # Save the DataFrame to CSV
            csv_data = day_wide.to_csv(index=False)
            blob_name = f"{PROJECT_PREFIX}/rainfall/gpm_data/rainfall_data/output_hhr_processed/{typ}_gridstats_{stats}.csv"

            # Upload the CSV to blob storage
            blob.upload_blob_data(
                blob_name=blob_name, data=csv_data, prod_dev=prod_dev
            )


def compute_stats(prod_dev="dev"):
    # Load metadata
    typhoon_metadata = load_metadata()

    # Ensure dates can be converted to datetime
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
    before_landfall_h = 72  # hours before landfall
    after_landfall_h = 72  # hours after landfall

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
            end_date = landfall + dt.timedelta(hours=after_landfall_h)
            start_date = landfall - dt.timedelta(hours=before_landfall_h)

            # Loading the data
            processed_file_path = f"{PROJECT_PREFIX}/rainfall/input_dir/gpm_data/rainfall_data/output_hhr_processed/{typ}_gridstats_{stats}.csv"
            df_rainfall = blob.load_csv(processed_file_path)

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

        # Save the DataFrame to CSV and upload to blob storage
        csv_data = df_rainfall_final.to_csv(index=False)
        output_blob_name = f"{PROJECT_PREFIX}/rainfall/output_dir/rainfall_data_rw_{stats}.csv"
        blob.upload_blob_data(
            blob_name=output_blob_name, data=csv_data, prod_dev=prod_dev
        )


if __name__ == "__main__":
    # Download data
    download_rainfall_data()
    # Create datasets
    create_rainfall_dataset()
    # Stats at grid level
    compute_stats()
