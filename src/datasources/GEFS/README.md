# GEFS rainfall data

```create_rainfall_dataset(metadata)``` uses NOMADS real-time rainfall data and compute the max rainfall accumulated in 6h and 24h periods for every grid cell. We considered 2 spatial methods: *mean* and *max* accumulated rainfall. By default, the output is a .csv of rainfall data for each wind event event that takes place in the region of interest defined in the. The user can load any metadata of events from any source.

The parameter **metadata** should be a Pandas Dataframe with 3 columns:

- 'event' (name or id of the event),
- 'start_date' (datetime format),
- 'end_date' (datetime format)


The fuction generates a csv with rainfall information at grid level. One can load the dataset by calling the function ```load_rainfall_data(date)```. By default, today's date is considered.
