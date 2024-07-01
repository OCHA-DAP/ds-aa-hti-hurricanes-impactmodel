#!/usr/bin/env python3
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from bs4 import BeautifulSoup
from rasterstats import zonal_stats

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"


def download_gefs_data(download_folder):
    # Calculate the current date
    today = datetime.now().strftime("%Y%m%d")

    # Generate folder names for 18, 12, 06, and 00 h
    folder_names = [f"{today}/18", f"{today}/12", f"{today}/06", f"{today}/00"]

    for folder in folder_names:
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.{folder}/prcp_bc_gb2/"
        response = requests.get(url)

        if response.status_code == 200:
            print("Downloading ", url)
            soup = BeautifulSoup(response.text, "html.parser")
            links = [
                a["href"]
                for a in soup.find_all("a")
                if a["href"].startswith("geprcp.t")
                and a["href"][:-7].endswith("bc_")
            ]

            if links:
                for link in links:
                    link_url = url + link
                    file_name = link.split("/")[-1]
                    file_path = os.path.join(download_folder, file_name)

                    download_response = requests.get(link_url)

                    with open(file_path, "wb") as f:
                        f.write(download_response.content)
                return
            break
        else:
            print("prcp_gb2 not found for today.")


def create_metadata(df_windfield):
    # Focus on Region of interest (ROI)
    roi_forecast = df_windfield[df_windfield.in_roi == True]

    # Group the DataFrame by the 'unique_id' column
    grouped = roi_forecast.groupby("unique_id")

    # Create a list of Forecasts DataFrames, one for each unique_id
    list_forecast = [group for name, group in grouped]

    # Metadata of events
    event_metadata = pd.DataFrame(
        {
            "event": roi_forecast.unique_id.unique(),
            "start_date": [df.time_init.iloc[0] for df in list_forecast],
            "end_date": [df.time_end.iloc[0] for df in list_forecast],
        }
    )
    return event_metadata


def create_rainfall_dataset(event_metadata, PROJECT_PREFIX=PROJECT_PREFIX):
    # Download today's forecasts
    today = datetime.now().strftime("%Y%m%d")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        download_folder = Path(temp_dir) / "NOMADS" / today / "input_files"
        os.makedirs(download_folder, exist_ok=True)
        download_gefs_data(download_folder)

        # Load grid cells
        grid = blob.load_grid(complete=False)
        grid["id"] = grid["id"].astype(int)

        # Load files
        gefs_folder_path = download_folder

        # Output folder
        processed_output_dir_grid = (
            Path(temp_dir) / "NOMADS" / today / "output_processed_bygrid"
        )
        os.makedirs(processed_output_dir_grid, exist_ok=True)

        processed_output_dir = (
            Path(temp_dir) / "NOMADS" / today / "output_processed"
        )
        os.makedirs(processed_output_dir, exist_ok=True)

        # Max and mean rainfall by grid

        # List of statistics to compute
        stats_list = ["mean", "max"]
        grid_list = []
        # Create a loop for running through all GEFS files
        for gefs_file in sorted(os.listdir(gefs_folder_path)):
            if gefs_file.startswith("geprcp"):
                gefs_file_path = Path(gefs_folder_path) / gefs_file
                input_raster = rasterio.open(gefs_file_path)
                array = input_raster.read(1)

                # Compute statistics for grid cells
                summary_stats = zonal_stats(
                    grid,
                    array,
                    stats=stats_list,
                    nodata=-999,  # There's no specificaiton on this, so we invent a number
                    all_touched=True,
                    affine=input_raster.transform,
                )

                grid_stats = pd.DataFrame(summary_stats)

                # Change values to mm/hr
                if gefs_file[:-5].endswith("06"):
                    grid_stats[stats_list] /= 6  # For 6h accumulation data
                else:
                    grid_stats[stats_list] /= 24  # For 24h accumulation data

                # Set appropriate date and time information
                forecast_hours = int(
                    gefs_file.split(".")[-1][-3:]
                )  # Extract forecast hours from the filename
                release_hour = int(
                    gefs_file.split(".")[1][1:3]
                )  # Extract release hour from the filename
                release_date = datetime.now().replace(
                    hour=release_hour, minute=0, second=0, microsecond=0
                )  # Set release date
                forecast_date = release_date + timedelta(
                    hours=forecast_hours
                )  # Calculate forecast date

                # Merge grid statistics with grid data
                grid_merged = pd.concat([grid, grid_stats], axis=1)
                # Set appropriate date and time information (modify as per GEFS data format)
                grid_merged["date"] = forecast_date.strftime(
                    "%Y%m%d%H"
                )  # Format date as string
                grid_merged = grid_merged[["id", "max", "mean", "date"]]
                grid_list.append(grid_merged)

                # Save the processed data. Name file by time and put in folder 06 or 24 regarding rainfall accumulation.
                grid_date = forecast_date.strftime("%Y%m%d%H")
                if gefs_file[:-5].endswith("06"):
                    # Set out dir
                    outdir = processed_output_dir_grid / "06"
                    os.makedirs(outdir, exist_ok=True)
                else:
                    # Set out dir
                    outdir = processed_output_dir_grid / "24"
                    os.makedirs(outdir, exist_ok=True)

                grid_merged.to_csv(
                    outdir / f"{grid_date}_gridstats.csv",
                    index=False,
                )

        # Create input rainfall dataset
        for folder in ["06", "24"]:
            list_df = []
            for file in sorted(os.listdir(processed_output_dir_grid / folder)):
                file_path = processed_output_dir_grid / folder / file
                df_aux = pd.read_csv(file_path)
                # Put 0 in nans values
                df_aux = df_aux.fillna(0)
                list_df.append(df_aux)

            final_df = pd.concat(list_df)
            # Convert the "date" column to datetime type
            final_df["date"] = pd.to_datetime(
                final_df["date"], format="%Y%m%d%H"
            )

            # Separate DataFrames for "mean" and "max" statistics
            df_pivot_mean = final_df.pivot_table(
                index=["id"], columns="date", values="mean"
            )
            df_pivot_max = final_df.pivot_table(
                index=["id"], columns="date", values="max"
            )

            # Flatten the multi-level column index
            df_pivot_mean.columns = [
                f"{col.strftime('%Y-%m-%d %H:%M:%S')}"
                for col in df_pivot_mean.columns
            ]
            df_pivot_max.columns = [
                f"{col.strftime('%Y-%m-%d %H:%M:%S')}"
                for col in df_pivot_max.columns
            ]

            # Reset the index
            df_pivot_mean.reset_index(inplace=True)
            df_pivot_max.reset_index(inplace=True)

            # Save .csv
            outdir = processed_output_dir / folder
            os.makedirs(outdir, exist_ok=True)
            df_pivot_mean.to_csv(
                outdir / str("gridstats_" + "mean" + ".csv"),
                index=False,
            )
            df_pivot_max.to_csv(
                outdir / str("gridstats_" + "max" + ".csv"),
                index=False,
            )

        # Compute 6h and 24h max rainfall
        events = event_metadata.event.to_list()
        time_frame_24 = 4  # in 6 hour periods
        time_frame_6 = 1  # in 6 hour periods

        for stats in ["mean", "max"]:
            df_rainfall_final = pd.DataFrame()
            for event in events:
                # Getting event info
                df_info = event_metadata[event_metadata["event"] == event]

                # End date is the end date of the forecast
                end_date = pd.to_datetime(df_info["end_date"].iloc[0])
                # Start date is starting day of the forecast
                start_date = pd.to_datetime(df_info["start_date"].iloc[0])

                # Loading the data (6h and 24h accumulation)
                df_rainfall_06 = pd.read_csv(
                    processed_output_dir
                    / "06"
                    / str("gridstats_" + stats + ".csv")
                )
                df_rainfall_24 = pd.read_csv(
                    processed_output_dir
                    / "24"
                    / str("gridstats_" + stats + ".csv")
                )

                # Focus on event dates (df_rainfall_06.iloc[3:] and df_rainfall_24.iloc[1:] have the same columns)
                available_dates_t = [
                    date
                    for date in df_rainfall_24.columns[1:]
                    if (pd.to_datetime(date) >= start_date)
                    & (pd.to_datetime(date) < end_date)
                ]
                # Restrict to event dates
                df_rainfall_06 = df_rainfall_06[["id"] + available_dates_t]
                df_rainfall_24 = df_rainfall_24[["id"] + available_dates_t]

                # Create new dataframe
                df_mean_rainfall = pd.DataFrame({"id": df_rainfall_24["id"]})

                df_mean_rainfall["rainfall_max_6h"] = (
                    df_rainfall_06.iloc[:, 1:]
                    .T.rolling(time_frame_6)
                    .mean()
                    .max(axis=0)
                )

                df_mean_rainfall["rainfall_max_24h"] = (
                    df_rainfall_24.iloc[:, 1:]
                    .T.rolling(time_frame_24)
                    .mean()
                    .max(axis=0)
                )
                df_rainfall_single = df_mean_rainfall[
                    [
                        "id",
                        "rainfall_max_6h",
                        "rainfall_max_24h",
                    ]
                ]
                df_rainfall_single["event"] = event
                df_rainfall_final = pd.concat(
                    [df_rainfall_final, df_rainfall_single]
                )

            # Saving the final rainfall dataset to blob storage
            today_just_date = datetime.now().strftime("%Y%m%d")
            final_path = f"{PROJECT_PREFIX}/rainfall/GEFS/{today_just_date}/rainfall_data_rw_{stats}.csv"
            csv_data = df_rainfall_final.to_csv(index=False)
            blob.upload_blob_data(blob_name=final_path, data=csv_data)


# Load data
# Calculate the current date
today = datetime.now().strftime("%Y%m%d")
def load_rainfall_data(date=today):
    # We are just interested in the spatial mean of the accmulated rainfall by grid
    df_rain = blob.load_csv(
        f"{PROJECT_PREFIX}/rainfall/GEFS/{date}/rainfall_data_rw_mean.csv"
    )
    return df_rain
