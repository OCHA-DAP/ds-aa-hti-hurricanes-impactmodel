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
shp_input_dir = base_dir / "02_housing_damage/input/"
grid_input_dir = base_dir / "02_housing_damage/output/"
output_dir = base_dir / "06_settlement/output/"
os.makedirs(output_dir, exist_ok=True)

# Load Shapefile
shp = gpd.read_file(shp_input_dir / "shapefile_hti_fixed.gpkg")
shp = shp.to_crs("EPSG:4326")

# Load grid cells
grid = gpd.read_file(grid_input_dir / "hti_0.1_degree_grid_land_overlap.gpkg")

# Load ids of municipalities
ids_mun = pd.read_csv(
    base_dir / "02_housing_damage/input/grid_municipality_info.csv"
)
ids_mun.ADM1_EN = ids_mun.ADM1_EN.str.upper()

# Load impact data
df_impact = pd.read_csv(shp_input_dir / "impact_data_clean_hti.csv")

# Load population data
grid_pop_df = pd.read_csv(output_dir / "hti_population_data.csv")


# Add Total pop by ADM1/ADM2 region feature to impact dataset
# Although this is irrelevant, it's here for completeness since it's relevent
# for other grid disaggregation definitions.
def add_pop_info_to_impact_data(grid_pop_df):
    # by ADMIN1
    df_merge_adm1 = (
        grid_pop_df.merge(ids_mun, left_on="id", right_on="id")[
            ["total_pop", "ADM1_PCODE"]
        ]
        .groupby("ADM1_PCODE")
        .sum()
        .reset_index(drop=False)
    )

    # by ADMIN2
    df_merge_adm2 = (
        grid_pop_df.merge(ids_mun, left_on="id", right_on="id")[
            ["total_pop", "ADM2_PCODE"]
        ]
        .groupby("ADM2_PCODE")
        .sum()
        .reset_index(drop=False)
    )

    # Add information of total population at ADM1/ADM2 level to the impact data
    df_impact_plus = df_impact.merge(
        df_merge_adm1[["total_pop", "ADM1_PCODE"]], on="ADM1_PCODE", how="left"
    ).rename({"total_pop": "total_pop_adm1"}, axis=1)
    df_impact_plus = df_impact_plus.merge(
        df_merge_adm2[["total_pop", "ADM2_PCODE"]], on="ADM2_PCODE", how="left"
    ).rename({"total_pop": "total_pop_adm2"}, axis=1)

    return df_impact_plus


# Impact data to grid level + NO WEATHER CONSTRAINTS
def impact_to_grid(grid_pop_df):
    # Load impact data + information
    df_impact_plus = add_pop_info_to_impact_data(grid_pop_df=grid_pop_df)
    pop_grid = grid_pop_df.merge(ids_mun, on="id")[
        ["id", "total_pop", "ADM1_PCODE", "ADM2_PCODE"]
    ]
    # Merge impact and population data
    pop_damage_merged_adm1 = df_impact_plus.merge(pop_grid, on="ADM1_PCODE")
    pop_damage_merged_adm1 = pop_damage_merged_adm1[
        ["typhoon_name", "Year", "id", "total_pop", "affected_population"]
    ].drop_duplicates()

    """
    Sometimes, for ADM2 events that we force them to be ADM1, we end up having 2 possible values of damage for the same grid cell:
        1- Either we have damage because we originally had damage at ADM1
        2- We originally didnt had damage at ADM2, so we still have this

    The solution for keeping things at ADM1 is to keep the ADM1 damage.
    Thats why we .drop_duplicates(subset='id', keep='last') after .sort_values('perc_aff_pop_grid')
    """
    df_events = pop_damage_merged_adm1[
        ["typhoon_name", "Year"]
    ].drop_duplicates()
    pop_damage_merged_adm1_fixed = pd.DataFrame()
    for typhoon, year in zip(df_events.typhoon_name, df_events.Year):
        # Select event
        df_event = pop_damage_merged_adm1[
            (pop_damage_merged_adm1.typhoon_name == typhoon)
            & (pop_damage_merged_adm1.Year == year)
        ]
        df_event = df_event.sort_values("affected_population").drop_duplicates(
            subset="id", keep="last"
        )
        # Append data
        pop_damage_merged_adm1_fixed = pd.concat(
            [pop_damage_merged_adm1_fixed, df_event]
        )

    # Events
    df_events = pop_damage_merged_adm1_fixed[
        ["typhoon_name", "Year"]
    ].drop_duplicates()

    # Iterate for every event
    impact_data_grid_no_weather = pd.DataFrame()
    for typhoon, year in zip(df_events.typhoon_name, df_events.Year):
        # Select event
        df_event = pop_damage_merged_adm1_fixed[
            (pop_damage_merged_adm1_fixed.typhoon_name == typhoon)
            & (pop_damage_merged_adm1_fixed.Year == year)
        ]
        df_event_dmg_with_pop = df_event[
            (df_event.total_pop > 1) & (df_event.affected_population != 0)
        ].copy()
        ids_dmg = df_event_dmg_with_pop.id
        # Total pop of country
        TOTAL_POP = df_event.total_pop.sum()
        # Total pop of region affected
        TOTAL_POP_REG = df_event_dmg_with_pop.total_pop.sum()

        # Perc of total pop affected in the area affected
        perc_dmg = (
            df_event_dmg_with_pop.affected_population.unique()
            / df_event_dmg_with_pop.total_pop.sum()
        )

        # If, for some reason, there are >100% dmg in some cells, set the dmg to 100%
        if perc_dmg > 1:
            perc_dmg = 1

        # Total pop affected by grid
        df_event_dmg_with_pop.loc[:, "affected_pop_grid"] = (
            df_event_dmg_with_pop.total_pop * perc_dmg
        )
        # % of affection with respect to the total pop of the country
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_country"] = (
            100 * df_event_dmg_with_pop.affected_pop_grid / TOTAL_POP
        )
        # % of affection with respect to the total pop of the region affected
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_region"] = (
            100 * df_event_dmg_with_pop.affected_pop_grid / TOTAL_POP_REG
        )
        # % of affection with respect to the total pop of grid
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_grid"] = (
            100
            * df_event_dmg_with_pop.affected_pop_grid
            / df_event_dmg_with_pop.total_pop
        )

        df_event = df_event.merge(df_event_dmg_with_pop, how="left").fillna(0)
        impact_data_grid_no_weather = pd.concat(
            [impact_data_grid_no_weather, df_event]
        )

    # No Weather constraints
    impact_data_grid_no_weather.to_csv(
        grid_input_dir / "impact_data_grid_step_disaggregation_no_weather.csv",
        index=False,
    )


def load_weather_features():
    # Load windspeed
    df_wind = pd.read_csv(
        base_dir / "01_windfield/windfield_data_hti_overlap.csv"
    )
    df_wind = df_wind.rename(
        {"track_id": "sid", "grid_point_id": "id"}, axis=1
    )
    df_wind["typhoon"] = df_wind["typhoon_name"] + df_wind[
        "typhoon_year"
    ].astype("str")

    # Load rainfall
    df_rain = pd.read_csv(
        base_dir / "03_rainfall/output/rainfall_data_rw_mean.csv"
    )

    # Merge
    df_weather = df_wind.merge(df_rain, on=["typhoon", "id"])
    return df_weather


def impact_to_grid_weather_constraints(grid_pop_df):
    # Load impact data + information
    df_impact_plus = add_pop_info_to_impact_data(grid_pop_df=grid_pop_df)
    pop_grid = grid_pop_df.merge(ids_mun, on="id")[
        ["id", "total_pop", "ADM1_PCODE", "ADM2_PCODE"]
    ]
    # Merge impact and population data
    pop_damage_merged_adm1 = df_impact_plus.merge(pop_grid, on="ADM1_PCODE")
    pop_damage_merged_adm1 = pop_damage_merged_adm1[
        ["typhoon_name", "Year", "id", "total_pop", "affected_population"]
    ].drop_duplicates()

    """
    Sometimes, for ADM2 events that we force them to be ADM1, we end up having 2 possible values of damage for the same grid cell:
        1- Either we have damage because we originally had damage at ADM1
        2- We originally didnt had damage at ADM2, so we still have this

    The solution for keeping things at ADM1 is to keep the ADM1 damage.
    Thats why we .drop_duplicates(subset='id', keep='last') after .sort_values('perc_aff_pop_grid')
    """
    df_events = pop_damage_merged_adm1[
        ["typhoon_name", "Year"]
    ].drop_duplicates()
    pop_damage_merged_adm1_fixed = pd.DataFrame()
    for typhoon, year in zip(df_events.typhoon_name, df_events.Year):
        # Select event
        df_event = pop_damage_merged_adm1[
            (pop_damage_merged_adm1.typhoon_name == typhoon)
            & (pop_damage_merged_adm1.Year == year)
        ]
        df_event = df_event.sort_values("affected_population").drop_duplicates(
            subset="id", keep="last"
        )

        # Append data
        pop_damage_merged_adm1_fixed = pd.concat(
            [pop_damage_merged_adm1_fixed, df_event]
        )

    # Apply weather constraints
    df_weather = load_weather_features()
    pop_dmg_wind_aux = pop_damage_merged_adm1_fixed.merge(df_weather)
    # Just affecting events
    pop_dmg_wind_aux = pop_dmg_wind_aux[
        pop_dmg_wind_aux.affected_population != 0
    ]
    # Just high windspeed events OR events with high precipitation
    THRES_W = 20  # m/s
    THRES_R = 30  # mm (daily)
    pop_dmg_wind_aux = pop_dmg_wind_aux[
        (pop_dmg_wind_aux.wind_speed >= THRES_W)
        | (pop_dmg_wind_aux.rainfall_max_24h >= THRES_R)
    ]
    pop_dmg_wind_aux = pop_dmg_wind_aux.reset_index(drop=True)[
        pop_damage_merged_adm1_fixed.columns
    ]

    # Events
    df_events = pop_dmg_wind_aux[["typhoon_name", "Year"]].drop_duplicates()

    # Iterate for every event
    impact_data_grid = pd.DataFrame()
    for typhoon, year in zip(df_events.typhoon_name, df_events.Year):
        # Select event
        df_event = pop_dmg_wind_aux[
            (pop_dmg_wind_aux.typhoon_name == typhoon)
            & (pop_dmg_wind_aux.Year == year)
        ]
        df_event_dmg_with_pop = df_event[
            (df_event.total_pop > 1) & (df_event.affected_population != 0)
        ].copy()
        ids_dmg = df_event_dmg_with_pop.id
        # Total pop of country
        TOTAL_POP = df_event.total_pop.sum()
        # Total pop of region affected
        TOTAL_POP_REG = df_event_dmg_with_pop.total_pop.sum()

        # Perc of total pop affected in the area affected
        perc_dmg = (
            df_event_dmg_with_pop.affected_population.unique()
            / df_event_dmg_with_pop.total_pop.sum()
        )
        # If, for weather constrains, there are >100% dmg in some cells, set the dmg to 100%
        if perc_dmg > 1:
            perc_dmg = 1
        # Total pop affected by grid
        df_event_dmg_with_pop.loc[:, "affected_pop_grid"] = (
            df_event_dmg_with_pop.total_pop * perc_dmg
        )
        # % of affection with respect to the total pop of the country
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_country"] = (
            100 * df_event_dmg_with_pop.affected_pop_grid / TOTAL_POP
        )
        # % of affection with respect to the total pop of the region affected
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_region"] = (
            100 * df_event_dmg_with_pop.affected_pop_grid / TOTAL_POP_REG
        )
        # % of affection with respect to the total pop of the grid
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_grid"] = (
            100
            * df_event_dmg_with_pop.affected_pop_grid
            / df_event_dmg_with_pop.total_pop
        )

        df_event = df_event.merge(df_event_dmg_with_pop, how="left").fillna(0)
        impact_data_grid = pd.concat([impact_data_grid, df_event])

    # Return to the complete dataset (with non-damaging regions by event)
    impact_data_grid = impact_data_grid.merge(
        pop_damage_merged_adm1_fixed, how="right"
    )[impact_data_grid.columns].fillna(0)

    # Save it to a csv
    impact_data_grid.to_csv(
        grid_input_dir / "impact_data_grid_step_disaggregation_weather.csv",
        index=False,
    )


if __name__ == "__main__":
    # Impact to grid level
    impact_to_grid(grid_pop_df=grid_pop_df)
    # Impact to grid level + weather_constraints
    impact_to_grid_weather_constraints(grid_pop_df=grid_pop_df)
