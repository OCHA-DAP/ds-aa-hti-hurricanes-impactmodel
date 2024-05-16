# Topography variables

We get terrain altitude data from [SRTM](https://dwtkns.com/srtm30m/). Here we need to register, select the desired location and download data. The data consists of a group of .hdg files.

We import each .hdg file, merge it into a single .tif file and compute mean altitude, mean slope and mean ruggedness measures at grid level.  Additionally, using the shapefile, we compute the length of the coastline for each grid cell and we created a binary variable *with_grid* that is 1 if the grid has a coastline or 0 if not.

Output: dataset of mean altitude, mean slope, mean_ruggedness, coast length and with_coast at grid level.
