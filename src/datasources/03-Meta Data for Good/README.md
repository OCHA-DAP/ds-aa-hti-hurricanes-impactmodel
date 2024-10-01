# Population data

- We download and aggregate to grid level the population data (number of people by grid cell) -from ```https://data.humdata.org/dataset/haiti-high-resolution-population-density-maps-demographic-estimates?```-

- We also disaggregate the impact data to grid level using a simple a approach: the density of impact (people affected over the total population of the area reported as affected) is constant along the affected area. So each grid cell in the affected area has a fix value of % of affection.

- In order to deal with bad reported data or data at a low level, we also considered applying weather constraints. So, the affected area reduces to affected area + certain conditions. The conditions imposed are: windspeed >= 20m/s OR rainfall_24h >= 30mm.
