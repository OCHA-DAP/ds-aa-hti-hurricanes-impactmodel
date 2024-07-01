# Windspeed features

We download track information from IbTracks and we compute the wind_speed and track_distance features at grid level.
Some considerations:

-  We create a 'balanced' dataset: equal number of impactful events and non-impactful events. For this, we choose the closest events to the country (using the ```hti_distances.csv``` dataset) and downloaded the track information for each one of them.
-  We perform interpolation between each coordinate (lat, lom, time) and weather related variable (central pressure, atmospheric pressure, wind_speed) in order to 'smooth' the tracks.
-  The output is at grid level (grid defined in the utils folder).
-  We also gathered the landfall information (if available) and created the ```typhoons.csv``` table with information about start_time, end_time and landfall_time of each event. If the event didn't make landfall, we define the landfall as the closest point (lan, lat, time) to the land.
