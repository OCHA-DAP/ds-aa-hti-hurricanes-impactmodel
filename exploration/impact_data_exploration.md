```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pathlib import Path
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

# Load shapefile

# Path to the extracted folder containing the shapefiles
extracted_folder = input_dir / "hti_adm_cnigs_20181129"

# Specify the filename of the shapefile associated with adm2
shapefile_adm2 = extracted_folder / "hti_admbnda_adm2_cnigs_20181129.shp"

# Read the shapefile into a GeoDataFrame
shp = gpd.read_file(shapefile_adm2)
shp = shp.to_crs('EPSG:4326')


# Load impact data
df_impact = pd.read_csv(input_dir / "impact_data_hti.csv")
```


```python
df_impact
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
      <th>affected_population</th>
      <th>Year</th>
      <th>sid</th>
      <th>typhoon_name</th>
      <th>affected_adm1_regions</th>
      <th>affected_adm2_regions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.0</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>Lili</td>
      <td>['Artibonite', 'Centre', 'Nord', 'Nord Est', '...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6500.0</td>
      <td>2004</td>
      <td>2004247N10332</td>
      <td>Ivan</td>
      <td>[]</td>
      <td>['Cap Haitien', 'Cayes']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>315594.0</td>
      <td>2004</td>
      <td>2004258N16300</td>
      <td>Jeanne</td>
      <td>['Artibonite', 'Centre', 'Nord Ouest', 'Sud']</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15036.0</td>
      <td>2005</td>
      <td>2005186N12299</td>
      <td>Hurricane "Dennis"</td>
      <td>['Ouest', 'Sud', 'Sud Est', 'Grande Anse', 'Ni...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750.0</td>
      <td>2005</td>
      <td>2005192N11318</td>
      <td>Emily</td>
      <td>[]</td>
      <td>['Saint-Marc']</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10000.0</td>
      <td>2005</td>
      <td>2005275N19274</td>
      <td>Stan</td>
      <td>[]</td>
      <td>['Dessalines', 'Saint-Marc']</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2192.0</td>
      <td>2005</td>
      <td>2005296N16293</td>
      <td>Alpha</td>
      <td>[]</td>
      <td>['Gros Morne', 'Hinche', 'Leogane', 'Port-Au-P...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15000.0</td>
      <td>2006</td>
      <td>2006237N13298</td>
      <td>Ernesto</td>
      <td>['Artibonite', 'Ouest', 'Sud', 'Grande Anse', ...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3966.0</td>
      <td>2007</td>
      <td>2007225N12331</td>
      <td>Dean</td>
      <td>['Artibonite', 'Centre', 'Nord', 'Nord Est', '...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>108763.0</td>
      <td>2007</td>
      <td>2007297N18300</td>
      <td>Noel</td>
      <td>[]</td>
      <td>['Gonaives', 'Port-Au-Prince', 'Cayes', 'Jacmel']</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2352.0</td>
      <td>2007</td>
      <td>2007345N18298</td>
      <td>Olga</td>
      <td>['Nord', 'Nord Est', 'Nord Ouest']</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>220.0</td>
      <td>2008</td>
      <td>2008229N18293</td>
      <td>Tropical Storm "Fay"</td>
      <td>['Artibonite', 'Centre', 'Nord', 'Nord Est', '...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>73006.0</td>
      <td>2008</td>
      <td>2008238N13293</td>
      <td>Hurricane "Gustav"</td>
      <td>['Artibonite', 'Centre', 'Ouest', 'Sud', 'Sud ...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>48000.0</td>
      <td>2008</td>
      <td>2008241N19303</td>
      <td>Hurricane 'Hanna'</td>
      <td>['Nord', 'Sud', 'Sud Est', 'Nippes']</td>
      <td>['Gros Morne', 'Gonaives', 'Saint-Marc', 'Port...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>125050.0</td>
      <td>2008</td>
      <td>2008245N17323</td>
      <td>Hurricane Ike</td>
      <td>[]</td>
      <td>['Gonaives']</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5020.0</td>
      <td>2010</td>
      <td>2010302N09306</td>
      <td>Hurricane Tomas</td>
      <td>['Grande Anse']</td>
      <td>['Leogane', 'Port-Au-Prince', 'Cayes', 'Jacmel']</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1500.0</td>
      <td>2011</td>
      <td>2011214N15299</td>
      <td>Tropical storm "Emily"</td>
      <td>['Sud']</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1544.0</td>
      <td>2011</td>
      <td>2011233N15301</td>
      <td>Hurricane Irene</td>
      <td>['Nord']</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8007.0</td>
      <td>2012</td>
      <td>2012234N16315</td>
      <td>Hurricane Isaac</td>
      <td>['Artibonite', 'Ouest', 'Sud', 'Sud Est', 'Gra...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>201850.0</td>
      <td>2012</td>
      <td>2012296N14283</td>
      <td>Hurricane Sandy</td>
      <td>['Artibonite', 'Ouest', 'Sud', 'Sud Est', 'Gra...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1969.0</td>
      <td>2015</td>
      <td>2015237N14315</td>
      <td>Hurricane Erika</td>
      <td>[]</td>
      <td>['Gonaives', 'Croix-Des-Bouquets', 'Port-Au-Pr...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2100439.0</td>
      <td>2016</td>
      <td>2016273N13300</td>
      <td>Hurricane 'Matthew'</td>
      <td>['Nord Ouest', 'Ouest', 'Sud', 'Sud Est', 'Gra...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>40092.0</td>
      <td>2017</td>
      <td>2017242N16333</td>
      <td>Hurricane 'Irma'</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44175.0</td>
      <td>2020</td>
      <td>2020233N14313</td>
      <td>Hurricane 'Laura'</td>
      <td>['Artibonite', 'Centre', 'Nord', 'Nord Est', '...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>Hurricane 'Elsa'</td>
      <td>['Ouest', 'Sud', 'Sud Est', 'Grande Anse', 'Ni...</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



## Inspection and cleaning


```python
import ast

# Remove "Hurricane" and "Tropical cyclone" substrings from typhoon_name column
df_impact['typhoon_name'] = df_impact[
    'typhoon_name'
    ].str.upper().str.replace('HURRICANE ', '').str.replace('TROPICAL CYCLONE ', '').str.replace('TROPICAL STORM', '').str.replace('"','').str.replace("'",'')

# Literal eval
df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].astype('str').apply(ast.literal_eval)
df_impact['affected_adm2_regions'] = df_impact['affected_adm2_regions'].astype('str').apply(ast.literal_eval)
# Explode
df_impact = df_impact.explode('affected_adm1_regions').explode('affected_adm2_regions')

# Uppercase
df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].str.upper()
df_impact['affected_adm2_regions'] = df_impact['affected_adm2_regions'].str.upper()

# Fix spelling
df_impact['affected_adm2_regions'] = df_impact['affected_adm2_regions'].replace('CAYES', 'LES CAYES')
df_impact['affected_adm2_regions'] = df_impact['affected_adm2_regions'].replace('CAP HAITIEN', 'CAP-HAITIEN')
df_impact['affected_adm2_regions'] = df_impact['affected_adm2_regions'].replace("ANSE-D'AINAULT", "ANSE D'HAINAULT")

df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].replace("GRANDE ANSE", "GRANDE'ANSE")
df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].replace('NORD OUEST', 'NORD-OUEST')
df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].replace('NORD EST', 'NORD-EST')
df_impact['affected_adm1_regions'] = df_impact['affected_adm1_regions'].replace('SUD EST', 'SUD-EST')

```


```python
# Define a function to determine the event_level
def determine_event_level(row):
    if pd.isna(row['affected_adm2_regions']):
        return 'ADM1'
    else:
        return 'ADM2'

# Apply the function to create the 'event_level' column
df_impact['event_level'] = df_impact.apply(lambda row: determine_event_level(row), axis=1)
```


```python
df_impact.typhoon_name.unique()
```




    array(['LILI', 'IVAN', 'JEANNE', 'DENNIS', 'EMILY', 'STAN', 'ALPHA',
           'ERNESTO', 'DEAN', 'NOEL', 'OLGA', ' FAY', 'GUSTAV', 'HANNA',
           'IKE', 'TOMAS', ' EMILY', 'IRENE', 'ISAAC', 'SANDY', 'ERIKA',
           'MATTHEW', 'IRMA', 'LAURA', 'ELSA'], dtype=object)




```python
# Modify shapefile
shp['ADM2_EN'] = shp['ADM2_EN'].str.upper()
shp['ADM1_FR'] = shp['ADM1_FR'].str.upper()
```

Check matching between shapefile and impact data


```python
set(df_impact.affected_adm1_regions.unique()) - set(shp.ADM1_FR.unique())
```




    {nan}




```python
set(df_impact.affected_adm2_regions.unique()) - set(shp.ADM2_EN.unique())
```




    {nan}



All regions match!

Let's create a mapping for every adm2 and adm2 region


```python
df_map = shp[['ADM1_FR', 'ADM2_EN']].drop_duplicates()
df_map
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
      <th>ADM1_FR</th>
      <th>ADM2_EN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GRANDE'ANSE</td>
      <td>ABRICOTS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NORD</td>
      <td>ACUL DU NORD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NORD-OUEST</td>
      <td>ANSE-A-FOLEUR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SUD-EST</td>
      <td>ANSE-A-PITRE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NIPPES</td>
      <td>ANSE-A-VEAU</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>SUD</td>
      <td>TIBURON</td>
    </tr>
    <tr>
      <th>136</th>
      <td>SUD</td>
      <td>TORBECK</td>
    </tr>
    <tr>
      <th>137</th>
      <td>NORD-EST</td>
      <td>TROU DU NORD</td>
    </tr>
    <tr>
      <th>138</th>
      <td>NORD-EST</td>
      <td>VALLIERES</td>
    </tr>
    <tr>
      <th>139</th>
      <td>ARTIBONITE</td>
      <td>VERRETTES</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 2 columns</p>
</div>



Apply this to the damage dataset


```python
df_impact_new = df_impact.reset_index(drop=True).copy()
for index, row in df_impact_new.iterrows():
    # Fix IRMA
    if pd.isnull(row['affected_adm1_regions']) and pd.isnull(row['affected_adm2_regions']):
        adm1_regions = df_map.ADM1_FR.unique()
        adm2_regions = df_map.ADM2_EN.unique()

        df_impact_new.at[index, 'affected_adm1_regions'] = adm1_regions
        df_impact_new.at[index, 'affected_adm2_regions'] = adm2_regions
    # Fix nans in adm1 regions
    elif pd.isnull(row['affected_adm1_regions']):
        region = row['affected_adm2_regions']
        adm1_region = df_map.loc[df_map['ADM2_EN'] == region, 'ADM1_FR'].values
        df_impact_new.at[index, 'affected_adm1_regions'] = adm1_region[0] if len(adm1_region) > 0 else None
    # Fix nans in adm2 regions
    elif pd.isnull(row['affected_adm2_regions']):
        region = row['affected_adm1_regions']
        adm2_regions = df_map.loc[df_map['ADM1_FR'] == region, 'ADM2_EN'].tolist()
        df_impact_new.at[index, 'affected_adm2_regions'] = adm2_regions
    # Fix wrong mapping for adm1 regions
    elif not pd.isnull(row['affected_adm1_regions']) and not pd.isnull(row['affected_adm2_regions']):
        region = row['affected_adm2_regions']
        adm1_region = df_map.loc[df_map['ADM2_EN'] == region, 'ADM1_FR'].values
        df_impact_new.at[index, 'affected_adm1_regions'] = adm1_region[0] if len(adm1_region) > 0 else None

```

Add pcode info


```python
df_impact_exploded = df_impact_new.explode('affected_adm2_regions').explode('affected_adm1_regions')
```


```python
df_impact_pcodes = df_impact_exploded.merge(shp,
                          left_on=['affected_adm1_regions', 'affected_adm2_regions'],
                          right_on=['ADM1_FR', 'ADM2_EN'],
                          how='left')[df_impact.columns.to_list() + ['ADM1_PCODE', 'ADM2_PCODE']].dropna()
```

## Some maps

### ADM1 maps


```python
typhoons = df_impact.dropna(subset='affected_adm1_regions').typhoon_name.unique()
df_damage = df_impact.copy()
fig, ax = plt.subplots(2,4, figsize=(15,8))
ax = ax.flatten()
i=0
for t in typhoons[:8]:
    df_aux = df_damage[df_damage['typhoon_name'] == t]
    regions = df_aux['affected_adm1_regions'].to_list()
    shp['colors'] = 'lightblue'  # Default color for all provinces
    shp.loc[shp['ADM1_FR'].isin(regions), 'colors'] = 'red'  # Color for the specified provinces

    # Plot Vietnam regions with the specified colors
    shp.plot(ax=ax[i], color=shp['colors'], edgecolor='0.3', linewidth=0.2, legend=True, label='Region Colors')
    ax[i].set_title(t.upper())
    i+=1

plt.suptitle('ADM1 level events', size=20)
plt.tight_layout()
plt.show()
```



![png](impact_data_exploration_files/impact_data_exploration_21_0.png)



### ADM2 maps


```python
typhoons = df_impact.dropna(subset='affected_adm2_regions').typhoon_name.unique()
df_damage = df_impact.copy()
fig, ax = plt.subplots(2,4, figsize=(15,8))
ax = ax.flatten()
i=0
for t in typhoons[:8]:
    df_aux = df_damage[df_damage['typhoon_name'] == t]
    regions = df_aux['affected_adm2_regions'].to_list()
    shp['colors'] = 'lightblue'  # Default color for all provinces
    shp.loc[shp['ADM2_EN'].isin(regions), 'colors'] = 'red'  # Color for the specified provinces

    # Plot Vietnam regions with the specified colors
    shp.plot(ax=ax[i], color=shp['colors'], edgecolor='0.3', linewidth=0.2, legend=True, label='Region Colors')
    ax[i].set_title(t.upper())
    i+=1

plt.suptitle('ADM2 level events', size=20)
plt.tight_layout()
plt.show()
```



![png](impact_data_exploration_files/impact_data_exploration_23_0.png)



## Impact data complete (with non impacting regions per event)


```python
df_map
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
      <th>ADM1_FR</th>
      <th>ADM2_EN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GRANDE'ANSE</td>
      <td>ABRICOTS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NORD</td>
      <td>ACUL DU NORD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NORD-OUEST</td>
      <td>ANSE-A-FOLEUR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SUD-EST</td>
      <td>ANSE-A-PITRE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NIPPES</td>
      <td>ANSE-A-VEAU</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>SUD</td>
      <td>TIBURON</td>
    </tr>
    <tr>
      <th>136</th>
      <td>SUD</td>
      <td>TORBECK</td>
    </tr>
    <tr>
      <th>137</th>
      <td>NORD-EST</td>
      <td>TROU DU NORD</td>
    </tr>
    <tr>
      <th>138</th>
      <td>NORD-EST</td>
      <td>VALLIERES</td>
    </tr>
    <tr>
      <th>139</th>
      <td>ARTIBONITE</td>
      <td>VERRETTES</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 2 columns</p>
</div>




```python
# For each of the events, add non impacting regions
event_names = df_impact_pcodes.typhoon_name.unique()
df_map = shp[['ADM1_PCODE', 'ADM2_PCODE']].drop_duplicates()

# For every event
df_impact_complete = pd.DataFrame()
for event in event_names:
    dft = df_impact_pcodes[df_impact_pcodes.typhoon_name == event].copy()

    year = dft.Year.iloc[0]
    sid = dft.sid.iloc[0]
    level = dft.event_level.iloc[0]

    dft_aux = dft.merge(df_map, on=['ADM1_PCODE', 'ADM2_PCODE'], how='right')
    dft_aux['typhoon_name'] = event
    dft_aux['Year'] = year
    dft_aux['sid'] = sid
    dft_aux['event_level'] = level

    # Fill nans
    dft_aux['affected_population'].fillna(0, inplace=True)

    df_impact_complete = pd.concat([df_impact_complete, dft_aux])

df_impact_complete = df_impact_complete.drop(['affected_adm1_regions', 'affected_adm2_regions'], axis=1)
```

## Save it


```python
# Impact data (just impacting areas by event)
df_impact_pcodes.to_csv(input_dir / "impact_data_clean_hti.csv", index=False)

# Impact data (all areas)
df_impact_complete.to_csv(input_dir / "impact_data_clean_complete_hti.csv", index=False)

# Modified shapefile
shp.to_file(input_dir / "shapefile_hti_fixed.gpkg", driver="GPKG")
```
