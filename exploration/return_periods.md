```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pathlib import Path

from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
import geopandas as gpd

from pyextremes import get_extremes, get_return_periods
```


```python
input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/input/"
)
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/output/"
)
```

Load impact data


```python
impact_data = pd.read_csv(input_dir / "impact_data_clean_hti.csv")[
    ['typhoon_name', 'Year', 'sid','affected_population']
    ].drop_duplicates().sort_values('Year', ascending=True).reset_index(drop=True)

```

Load tracks


```python
# Download NECESARY tracks
sel_ibtracs = []
for track in impact_data.sid:
    sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))

# Get tracks
tc_tracks = TCTracks()
for track in sel_ibtracs:
    tc_track = track.get_track()
    tc_track.interp(
        time = pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30T")
    )
    tc_tracks.append(tc_track)
```

    2024-04-30 20:24:24,103 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:27,511 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:30,543 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:33,669 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:36,826 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:39,917 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:42,993 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:46,107 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:49,213 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:52,570 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:55,672 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:24:58,775 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:01,837 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:05,039 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:08,119 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:11,170 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:14,235 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:17,383 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:20,429 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:23,572 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:26,706 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:29,768 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:32,910 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:36,005 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-04-30 20:25:39,010 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.


Return period indicates duration of time (typically years) which corresponds to a probability that a given value (e.g. wind speed) would be exceeded at least once within a year.

This probability is called probability of exceedance and is related to return periods as 1/p where p is return period.


https://georgebv.github.io/pyextremes/user-guide/6-return-periods/

## Example


```python
# Example
time = tc_tracks.get_track()[0].time
windspeed = tc_tracks.get_track()[0].max_sustained_wind
data = pd.DataFrame({'time':time, 'windspeed':windspeed}).set_index('time')

# For the library to work (https://github.com/georgebv/pyextremes-notebooks/blob/master/notebooks/EVA%20basic.ipynb)
data = (
    data
    .sort_index(ascending=True)
    .astype(float)
    .dropna()
)
# To series
data = data.squeeze()
```


```python
# This is just the maximum recorded wind_speed
extremes = get_extremes(
    ts=data,
    method="BM",
    block_size="365.2425D",
)

# Using weibull method, based on shape and scale of wind_speed distribution
return_periods = get_return_periods(
    ts=data,
    extremes=extremes,
    extremes_method="BM",
    extremes_type="high",
    block_size="365.2425D",
    return_period_size="365.2425D",
    plotting_position="weibull",
)
extremes
return_periods
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>windspeed</th>
      <th>exceedance probability</th>
      <th>return period</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-10-02 20:13:00</th>
      <td>125.0</td>
      <td>0.5</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## get return periods


```python
# Total temporal data
total_data = pd.Series()
extremes = pd.Series()
for i in range(len(impact_data)):
    track = tc_tracks.get_track()[i]
    windspeed = np.array(track.max_sustained_wind)
    time = np.array(track.time)

    data = pd.DataFrame({'time':time, 'windspeed':windspeed}).set_index('time')
    data = data.sort_index(ascending=True).astype(float).dropna().squeeze()
    # Extreme point (max windspeed)
    extreme = get_extremes(
        ts=data,
        method="BM",
        block_size="365.2425D",
    )
    extremes = pd.concat([extremes, extreme])
    total_data = pd.concat([total_data, data])

total_data = total_data.sort_index(ascending=True)
```

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_13730/4068764564.py:17: FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
      extremes = pd.concat([extremes, extreme])
    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_13730/4068764564.py:18: FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
      total_data = pd.concat([total_data, data])



```python
# Return period
T = get_return_periods(
    ts=total_data,
    extremes=extremes,
    extremes_method="BM",
    extremes_type="high",
    block_size="30D",
    return_period_size="30D",
    plotting_position="weibull").rename({None:'windspeed'}, axis=1)

# Add event name and change index
T = T.reset_index(drop=False).rename(columns={'index': 'time'})
T['event'] = impact_data.typhoon_name.to_list() # After checking the order

# Sort by RP
T = T.sort_values('return period', ascending=False).reset_index(drop=True)
```


```python
fig, ax = plt.subplots(1,1, figsize=(10,5))
# Plot
ax.plot(T['windspeed'], T['return period'], 'o', alpha=0.4, label='TCs in Haiti')
ax.set_xlabel('Max Sustained Windspeed (m/s)', size=15)
ax.set_ylabel('Return Period (years)', size=15)

# Threshold plot
ax.hlines(y=3, xmin=40, xmax=160, linestyle='--', colors='red', label='Threshold')
ax.set_xlim([40,160])
ax.grid()

# Dictionary to store the number of events with the same return period
rp_counts = {}

# Annotate events
for idx, event in T.iterrows():
    if event['return period'] >= 3:
        rp = event['return period']
        rp_count = rp_counts.get(rp, 0)
        y_offset = rp_count * 10
        x_offset = rp_count * 10  # Adjust this value to change the spacing between event names
        ax.annotate(event['event'],
                    (event['windspeed'], event['return period']),
                    textcoords="offset points",
                    xytext=(x_offset,y_offset),
                    ha='center',
                    fontsize=8,
                    rotation=90)
        rp_counts[rp] = rp_count + 1


ax.legend()
plt.show()

```



![png](return_periods_files/return_periods_13_0.png)




```python
T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>windspeed</th>
      <th>exceedance probability</th>
      <th>return period</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-09-05 18:00:00</td>
      <td>155.0</td>
      <td>0.038462</td>
      <td>26.000000</td>
      <td>IRMA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-08-21 06:00:00</td>
      <td>150.0</td>
      <td>0.076923</td>
      <td>13.000000</td>
      <td>DEAN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-10-01 00:00:00</td>
      <td>145.0</td>
      <td>0.134615</td>
      <td>7.428571</td>
      <td>MATTHEW</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-09-11 18:00:00</td>
      <td>145.0</td>
      <td>0.134615</td>
      <td>7.428571</td>
      <td>IVAN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-07-17 00:00:00</td>
      <td>140.0</td>
      <td>0.192308</td>
      <td>5.200000</td>
      <td>EMILY</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008-08-30 22:00:00</td>
      <td>135.0</td>
      <td>0.230769</td>
      <td>4.333333</td>
      <td>GUSTAV</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005-07-08 12:00:00</td>
      <td>130.0</td>
      <td>0.288462</td>
      <td>3.466667</td>
      <td>DENNIS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-08-27 00:00:00</td>
      <td>130.0</td>
      <td>0.288462</td>
      <td>3.466667</td>
      <td>LAURA</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2002-10-02 20:13:00</td>
      <td>125.0</td>
      <td>0.365385</td>
      <td>2.736842</td>
      <td>LILI</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2008-09-04 06:00:00</td>
      <td>125.0</td>
      <td>0.365385</td>
      <td>2.736842</td>
      <td>IKE</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004-09-25 18:00:00</td>
      <td>105.0</td>
      <td>0.442308</td>
      <td>2.260870</td>
      <td>JEANNE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2011-08-24 12:00:00</td>
      <td>105.0</td>
      <td>0.442308</td>
      <td>2.260870</td>
      <td>IRENE</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012-10-25 05:25:00</td>
      <td>100.0</td>
      <td>0.500000</td>
      <td>2.000000</td>
      <td>SANDY</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2010-10-30 20:00:00</td>
      <td>85.0</td>
      <td>0.538462</td>
      <td>1.857143</td>
      <td>TOMAS</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2008-09-02 00:00:00</td>
      <td>75.0</td>
      <td>0.615385</td>
      <td>1.625000</td>
      <td>HANNA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2007-11-03 12:00:00</td>
      <td>75.0</td>
      <td>0.615385</td>
      <td>1.625000</td>
      <td>NOEL</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2021-07-02 18:00:00</td>
      <td>75.0</td>
      <td>0.615385</td>
      <td>1.625000</td>
      <td>ELSA</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2012-08-28 18:00:00</td>
      <td>70.0</td>
      <td>0.711538</td>
      <td>1.405405</td>
      <td>ISAAC</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2005-10-04 12:00:00</td>
      <td>70.0</td>
      <td>0.711538</td>
      <td>1.405405</td>
      <td>STAN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2006-08-27 06:00:00</td>
      <td>65.0</td>
      <td>0.769231</td>
      <td>1.300000</td>
      <td>ERNESTO</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008-08-19 18:00:00</td>
      <td>60.0</td>
      <td>0.807692</td>
      <td>1.238095</td>
      <td>FAY</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2007-12-11 18:00:00</td>
      <td>50.0</td>
      <td>0.846154</td>
      <td>1.181818</td>
      <td>OLGA</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2011-08-03 00:00:00</td>
      <td>45.0</td>
      <td>0.923077</td>
      <td>1.083333</td>
      <td>EMILY</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2015-08-27 06:00:00</td>
      <td>45.0</td>
      <td>0.923077</td>
      <td>1.083333</td>
      <td>ERIKA</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2005-10-23 00:00:00</td>
      <td>45.0</td>
      <td>0.923077</td>
      <td>1.083333</td>
      <td>ALPHA</td>
    </tr>
  </tbody>
</table>
</div>
