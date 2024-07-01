#!/usr/bin/env python3
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from climada.hazard import Centroids, Hazard, TCTracks, TropCyclone
from climada_petals.hazard import TCForecast
from shapely.geometry import LineString, Point, Polygon

from src.utils import blob

PROJECT_PREFIX = "ds-aa-hti-hurricanes"


def trigger(df_windfield):
    length = len(df_windfield[df_windfield.in_roi == True])  # In ROI
    if length == 0:
        return False
    else:
        return True


def create_windfield_dataset(thres=120, deg=3):
    # Calculate the current date
    today_just_date = datetime.now().strftime("%Y%m%d")

    # Constant
    DEG_TO_KM = 111.1

    # Load grid cells
    grids = blob.load_grid(complete=False)
    grids.geometry = grids.geometry.to_crs(grids.crs).centroid

    try:
        # Load ECMWF data
        tc_fcast = TCForecast()
        tc_fcast.fetch_ecmwf()

        # Modify each of the event based on threshold
        n_events = len(tc_fcast.data)
        # Threshold
        today = datetime.now()
        # Calculate the threshold datetime from the current date and time
        threshold_datetime = np.datetime64(today + timedelta(hours=thres))

        xarray_data_list = []
        for i in range(n_events):
            data_event = tc_fcast.data[i]
            # Elements to consider
            index_thres = len(
                np.where(np.array(data_event.time) < threshold_datetime)[0]
            )
            if index_thres > 4:  # Events with at least 4 datapoints
                data_event_thres = data_event.isel(time=slice(0, index_thres))
                xarray_data_list.append(data_event_thres)
            else:
                continue

        # Create TropCyclone class with modified data (Predict Windfield on centroids)
        cent = Centroids.from_geodataframe(grids)
        tc_fcast_mod = TCForecast(xarray_data_list)
        tc = TropCyclone.from_tracks(
            tc_fcast_mod,
            centroids=cent,
            store_windfields=True,
            intensity_thres=0,
        )

        # Create windfield dataset
        event_names = list(tc.event_name)

        # Define the boundaries for the Region Of Interest (ROI)
        xmin, xmax, ymin, ymax = -75 - deg, -71 + deg, 17 - deg, 21 + deg
        roi_polygon = Polygon(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        )

        df_windfield = pd.DataFrame()
        for i, intensity_sparse in enumerate(tc.intensity):
            # Get the windfield
            windfield = intensity_sparse.toarray().flatten()
            npoints = len(windfield)
            event_id = event_names[i]

            # Track distance
            tc_track = tc_fcast_mod.get_track()[i]
            points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
            tc_track_line = LineString(points)
            tc_track_distance = grids["geometry"].apply(
                lambda point: point.distance(tc_track_line) * DEG_TO_KM
            )

            # Basin
            basin = np.unique(tc_track.basin)

            # Adquisition Period
            time0 = np.unique(tc_track.time)[0]
            time1 = np.unique(tc_track.time)[-1]

            # Does it touch ROI borders?
            intersects_roi = tc_track_line.intersects(roi_polygon)

            # Add to DF
            df_to_add = pd.DataFrame(
                dict(
                    event_id_ecmwf=[event_id] * npoints,
                    unique_id=[i] * npoints,
                    basins=[basin.tolist()] * npoints,
                    time_init=[time0] * npoints,
                    time_end=[time1] * npoints,
                    in_roi=[intersects_roi] * npoints,
                    grid_point_id=grids["id"],
                    wind_speed=windfield,
                    track_distance=tc_track_distance,
                    geometry=grids.geometry,
                )
            )
            df_windfield = pd.concat(
                [df_windfield, df_to_add], ignore_index=True
            )

        # Save results if there are results
        if trigger(df_windfield=df_windfield):
            csv_data = df_windfield.to_csv(index=False)
            wind_dir = (
                f"{PROJECT_PREFIX}/windfield/ECMWF/{today_just_date}/wind_data.csv"
            )
            blob.upload_blob_data(wind_dir, csv_data)
            print("High windspeed detected in the region of interest")
            return True

        else:
            print("{}: Wind Trigger not activated".format(today_just_date))
            return False

    except Exception as e:
        print("ECMWF not responding")
        raise ValueError("ECMWF not responding") from e


# Load data
# Calculate the current date
today = datetime.now().strftime("%Y%m%d")

def load_windspeed_data(date=today):
    wind_dir = f"{PROJECT_PREFIX}/windfield/ECMWF/{date}/wind_data.csv"
    df_wind = blob.load_csv(wind_dir)
    return df_wind
