#!/usr/bin/env python3
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
from wind_functions import (
    add_interpolation_points,
    adjust_tracks,
    calculate_mean_for_neighbors,
    get_closest_point_index,
    windfield_to_grid,
)

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"

# input and output dir
input_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features"
)
output_dir = input_dir / "01_windfield"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

"""     Load grid data and shapefile        """
# Load grid-land overlap data
gdf = blob.load_grid_centroids(complete=False)
# Load all grid data (include oceans)
gdf_all = blob.load_grid_centroids(complete=True)

# Centroids
cent = Centroids.from_geodataframe(gdf)  # grid-land overlap
cent_all = Centroids.from_geodataframe(gdf_all)  # include oceans

# Load shapefile
shp = blob.load_shp()


"""     Load impact data       """


def load_impact_data():
    # House impact data / Pop impact data
    df_housing = blob.load_emdat()

    # List of typhoons with impact data > 0
    typhoons_df = (
        df_housing[["typhoon_name", "Year", "sid"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    impacting_events = typhoons_df.copy()
    impacting_events["affected_pop"] = True

    # List of events with no impact data
    df_no_impact = blob.load_hti_distances()
    df_no_impact = df_no_impact.rename({"name": "typhoon_name"}, axis=1)
    df_no_impact = df_no_impact[
        ~df_no_impact.sid.isin(impacting_events.sid.unique())
    ]  # Drop the impact subset
    df_no_impact["Year"] = df_no_impact.time.apply(lambda x: int(x[:4]))
    df_no_impact = (
        df_no_impact[df_no_impact.typhoon_name != "NOT_NAMED"][
            ["typhoon_name", "Year", "sid", "distance (m)"]
        ]
        .drop_duplicates(subset=["typhoon_name", "Year", "sid"])
        .reset_index(drop=True)
    )  # Dont consider these ones
    # There are 2300 events.. let just consider the ones thar are the closest to the land.
    # Also, to make a balanced dataset, let just consider the same number of events as the ones that caused damage
    non_impacting_events = df_no_impact.sort_values("distance (m)").iloc[
        : len(impacting_events)
    ]
    non_impacting_events = non_impacting_events.drop(
        "distance (m)", axis=1
    ).copy()
    non_impacting_events["affected_pop"] = False

    # Non impacting + impacting events
    all_events = (
        pd.concat([impacting_events, non_impacting_events])
        .sort_values(["Year", "typhoon_name"])
        .reset_index(drop=True)
    )
    return all_events, non_impacting_events


"""     Download and proccess tracks      """


def get_storm_tracks(all_events):
    # Constant
    DEG_TO_KM = 111.1  # Convert 1 degree to km

    sel_ibtracs = []
    for track in all_events.sid:
        sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))

    # Interpolation proccess with current data (Doesnt make a difference if the data has few datapoints)
    # obs: .interp(x0,x,f(x)) gives the position of x0 in the fitting of (x,f(x))
    # obs: daterange consider the track between certain intervals as discrete points instead of a continuous
    tc_tracks = TCTracks()
    for track in sel_ibtracs:
        tc_track = track.get_track()
        tc_track.interp(
            time=pd.date_range(
                tc_track.time.values[0], tc_track.time.values[-1], freq="30T"
            )
        )
        tc_tracks.append(tc_track)
    return tc_tracks


# Add interpolation points (smooth the track)
def proccess_storm_tracks(tc_tracks):
    tracks = TCTracks()
    for i in range(len(tc_tracks.get_track())):
        # Define relevant features
        track_xarray = tc_tracks.get_track()[i]
        time_array = np.array(track_xarray.time)
        time_step_array = np.array(track_xarray.time_step)
        lat_array = np.array(track_xarray.lat)
        lon_array = np.array(track_xarray.lon)
        max_sustained_wind_array = np.array(track_xarray.max_sustained_wind)
        central_pressure_array = np.array(track_xarray.central_pressure)
        environmental_pressure_array = np.array(
            track_xarray.environmental_pressure
        )
        r_max_wind_array = np.array(track_xarray.radius_max_wind)
        r_oci_array = np.array(track_xarray.radius_oci)

        # Define new variables
        # Interpolate every important data
        w = max_sustained_wind_array.copy()
        t = time_array.copy()
        t_step = time_step_array.copy()
        lat = lat_array.copy()
        lon = lon_array.copy()
        cp = central_pressure_array.copy()
        ep = environmental_pressure_array.copy()
        rmax = r_max_wind_array.copy()
        roci = r_oci_array.copy()

        # Define the number of points to add between each pair of data points
        num_points_between = 2

        # Add interpolation points to regulat variables
        new_w = add_interpolation_points(w, num_points_between)
        new_t_step = add_interpolation_points(t_step, num_points_between)
        new_lat = add_interpolation_points(lat, num_points_between)
        new_lon = add_interpolation_points(lon, num_points_between)
        new_cp = add_interpolation_points(cp, num_points_between)
        new_ep = add_interpolation_points(ep, num_points_between)
        new_rmax = add_interpolation_points(rmax, num_points_between)
        new_roci = add_interpolation_points(roci, num_points_between)

        # Add interpolation points to time variables
        timestamps = np.array(
            [date.astype("datetime64[s]").astype("int64") for date in t]
        )  # Convert to seconds
        new_t = add_interpolation_points(timestamps, num_points_between)
        new_t = [
            np.datetime64(int(ts), "s") for ts in new_t
        ]  # Back to datetime format

        # Define dataframe
        df_t = pd.DataFrame(
            {
                "MeanWind": new_w,
                "PressureOCI": new_ep,
                "Pressure": new_cp,
                "Latitude": new_lat,
                "Longitude": new_lon,
                "RadiusMaxWinds": new_rmax,
                "RadiusOCI": new_roci,
                "time_step": new_t_step,
                "basin": np.array(
                    [np.array(track_xarray.basin)[0]] * len(new_t)
                ),
                "forecast_time": new_t,
                "Category": track_xarray.category,
            }
        )

        # Define a custom id
        custom_idno = track_xarray.id_no
        custom_sid = track_xarray.sid
        name = track_xarray.name  # + ' interpolated'

        # Define track as climada likes it
        track = TCTracks()
        track.data = [
            adjust_tracks(
                df_t, name=name, custom_sid=custom_sid, custom_idno=custom_idno
            )
        ]

        # Tracks modified
        tracks.append(track.get_track())
    return tracks


# Convert DataFrame to CSV in-memory
def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()


def create_windfield_features(tracks, non_impacting_events):
    # TropCyclone class
    tc_all = TropCyclone.from_tracks(
        tracks, centroids=cent_all, store_windfields=True, intensity_thres=0
    )

    # Create grid-level windfield
    df_windfield_interpolated = windfield_to_grid(
        tc=tc_all, tracks=tracks, grids=gdf_all
    )

    # Overlap
    df_windfield_interpolated_overlap = df_windfield_interpolated[
        df_windfield_interpolated.grid_point_id.isin(gdf.id)
    ]

    # Assign affection boolean variable
    df_windfield_interpolated_overlap.loc[
        df_windfield_interpolated_overlap["track_id"].isin(
            non_impacting_events.sid.unique()
        ),
        "affected_pop",
    ] = False
    df_windfield_interpolated_overlap.loc[
        ~df_windfield_interpolated_overlap["track_id"].isin(
            non_impacting_events.sid.unique()
        ),
        "affected_pop",
    ] = True

    # Cast affected_pop column to boolean dtype
    df_windfield_interpolated_overlap[
        "affected_pop"
    ] = df_windfield_interpolated_overlap["affected_pop"].astype(bool)

    # Save csv
    csv_data = dataframe_to_csv_bytes(df_windfield_interpolated_overlap)
    blob.upload_blob_data(
        blob_name=PROJECT_PREFIX
        + "/windfield/output_dir/windfield_data_hti_overlap.csv",
        data=csv_data,
    )


"""     Metadata of the events      """


def create_metadata(tracks, all_events):
    df_metadata_fixed = pd.DataFrame()
    for i in range(len(tracks.data)):
        # Basics
        startdate = np.datetime64(np.array(tracks.data[i].time[0]), "D")
        enddate = np.datetime64(np.array(tracks.data[i].time[-1]), "D")
        name = tracks.data[i].name
        year = tracks.data[i].sid[:4]
        nameyear = name + year

        # For the landfall
        # Track path
        tc_track = tracks.get_track()[i]
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        track_points = gpd.GeoDataFrame(geometry=points)

        # Set crs
        track_points.crs = shp.crs

        try:
            # intersection --> Look for first intersection == landfall
            min_index = shp.sjoin(track_points)["index_right"].min()

            landfalldate = np.datetime64(
                np.array(tracks.data[i].time[min_index]), "D"
            )
            landfall_time = str(
                np.datetime64(np.array(tracks.data[i].time[min_index]), "s")
            ).split("T")[1]
        except:
            # No landfall situation --> Use closest point to shapefile
            closest_point_index = get_closest_point_index(track_points, shp)
            landfalldate = np.datetime64(
                np.array(tracks.data[i].time[closest_point_index]), "D"
            )
            landfall_time = str(
                np.datetime64(
                    np.array(tracks.data[i].time[closest_point_index]), "s"
                )
            ).split("T")[1]

        # Create df
        df_aux = pd.DataFrame(
            {
                "typhoon": [nameyear],
                "startdate": [startdate],
                "enddate": [enddate],
                "landfalldate": [landfalldate],
                "landfall_time": [landfall_time],
            }
        )
        df_metadata_fixed = pd.concat([df_metadata_fixed, df_aux])
    df_metadata_fixed = df_metadata_fixed.reset_index(drop=True)

    # Add affected pop information
    all_events_meta = all_events.copy()
    all_events_meta["typhoon"] = all_events_meta[
        "typhoon_name"
    ] + all_events_meta["Year"].astype("str")
    all_events_meta = all_events_meta[["typhoon", "affected_pop"]]

    # Merge with landfall information
    df_metadata_fixed_complete = df_metadata_fixed.merge(all_events_meta)

    # Save the DataFrame to CSV
    csv_data = dataframe_to_csv_bytes(df_metadata_fixed_complete)
    blob.upload_blob_data(
        blob_name=PROJECT_PREFIX + "/rainfall/input_dir/metadata_typhoons.csv",
        data=csv_data,
    )


if __name__ == "__main__":
    # Load impact data
    all_events, non_impacting_events = load_impact_data()
    # Get tracks
    tc_tracks = get_storm_tracks(all_events=all_events)
    # Proccess tracks
    tracks = proccess_storm_tracks(tc_tracks=tc_tracks)
    # Create features
    create_windfield_features(
        tracks=tracks, non_impacting_events=non_impacting_events
    )
    # Create metadata
    create_metadata(tracks=tracks, all_events=all_events)
