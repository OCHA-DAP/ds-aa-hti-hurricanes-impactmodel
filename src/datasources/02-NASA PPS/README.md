# Rainfall information

## Download stage

We basically download the rainfall data from [GPM](https://arthurhouhttps.pps.eosdis.nasa.gov/pub/gpmdata). We need to register for this. It's everything explained at the code.

Output: GPM files for each typhoon.

## Proccesing stage

We compute the mean and max rainfall measured, at grid level.

Output: .csv files for each typhoon containing stats information at grid level.

## Feature creation stage

We compute for every typhoon the max rainfall (mm/hr) at grid level using 2 time intervals: 6h and 24h.

Output: a dataset containing the grid_id, max_6h and max_24h measures for every grid cell and for every typhoon.
