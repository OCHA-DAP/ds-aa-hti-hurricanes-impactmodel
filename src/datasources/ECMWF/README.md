# ECMWF

We use the ECMWF forecast provider to track storms in a specific area defined by a polygon -region of interest- (ROI).


```create_windfield_dataset(thres=120, deg=3)``` loads ECMWF real-time forecasts, and merge it with the grid dataset that we have already defined. Also if the forecast event took place in the Region of interest, a new feature *in_roi* is set True. To define the region of interest, we create a polygon (a square) with a custom size and if any point of the forecast track falls into this square, we automatically set that forecast as region-related. The user can obviously play with the dimension of this square.

Parameters:

*thres*: The user can set a threshold in hours: this threshold let the user to just consider datapoints up to a certain time in hours starting from the collection datetime of the data. The default value is 120h. So every wind forecast that the ECMWF provides is cut to just consider wind paths that just contemplates datapoints values up to 120h. Also, a wind path is considered just if it has at least 4 datapoints.

*deg*: Degrees to consider to extend the Polygone (in evey direction). By default, a polygone with coordinates (long_min, long_max, lat_min, lat_max) = (-75, -71, 17, 21) is set. What *deg* does is basically a transformation (-75-deg, -71+deg, 17-deg, 21+deg).


If activity is happening in the ROI, we save the activity (windspeed, track_distance) transformed to grid level in a dataset. To load this dataset, we call the function ```load_windspeed_data(date)```. By default, we use today's date.
