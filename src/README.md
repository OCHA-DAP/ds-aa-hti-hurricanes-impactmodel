# Model for predicting affected population on Haiti based on historical impacting events

Here we apply a grid-based approach to predict TC impact on population. Some considerations:

1) We use a grid based approach: disaggregate every feature + the target variable to pre-defined grid cells.

2) We consider features such as
- topographical-based (elevation, slope and ruggedness of the terrain)
- weather-based (Max sustained 1-min windspeed, distance to the track, Mean -spatial- Max -temporal- accumulated rainfall on 6h and 24h periods)
- population related (total population and total number of buildings)
- vulnerability-based (International Wealth Index)

3) The target variable (affected population) is disaggregated to grid level using a simple a approach: the density of impact (people affected over the total population of the area reported as affected) is constant along the affected area. So each grid cell in the affected area has a fix value of % of affection. In order to deal with bad reported data or data at a low level, we also considered applying weather constraints. So, the affected area reduces to affected area + certain conditions. The conditions imposed are: windspeed >= 20m/s OR rainfall_24h >= 30mm.


4) We use a 2-stage XGBoost model. The same that was used in [1]. The predictions are at grid level. The results should be aggregated to some higher level (ADM0 is desirable). Based on LOOCV training, the model, after binarization (threshold based on the median of the distribution of ADM0-impact-reported-subset) of predictions and actual impact data, achieves an f1-score of 0.59 and an accuracy of 0.81. This is better than an ADM0 (non grid-based approach) model (with just weather features) which reports an f1-score of 0.46 and an accuracy of 0.7 after the same binarization.

5) We make predicitons of affected population using real-time weather related features.

How to replicate the model? --> Follow these steps:

- Create a global enviroment called STORM_DATA_DIR that points to the data folder. Data is in this Drive: https://drive.google.com/drive/folders/1xaZ_y45Goqs8rO0ICW3egMfBjiWAKSdE?usp=sharing

- Create/define the grid cells by running the *create_grid.py* script inside the **utils** folder.

- To get the features, run every .py file inside each datasource (see datasource folder).

- Merge the features by running the *merge_features.py* script inside the **utils** folder.

- Run *model.py*. The results/predicitons are stored in *analysis_hti/04_model_output_dataset/*



[1] Towards a global impact-based forecasting model for tropical cyclones. Kooshki Forooshani, Mersedeh and van den Homberg, Marc and Kalimeri, Kyriaki and Kaltenbrunner, Andreas and Mejova, Yelena and Milano, Leonardo and Ndirangu, Pauline and Paolotti, Daniela and Teklesadik, Aklilu and Turner, Monica L. Natural Hazards and Earth System Sciences, 2024.
