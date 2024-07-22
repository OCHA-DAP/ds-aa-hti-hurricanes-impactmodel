# Haiti Anticipatory Action: hurricanes - impact model

Here we create and define a model for prediciting impact (in the form of affected population) caused by hurricanes in Haiti.
The model is essentially a 2-stage-grid-based-XGBoost. This model was previously defined for the Philippines in [[1]](https://www.researchgate.net/publication/378179311_Towards_a_global_impact-based_forecasting_model_for_tropical_cyclones).

The output of the model is predictions for affected population at administrative 1 level.


## Directory structure

The code in this repository is organized as follows:

```shell

├── exploration   # Experimental work not intended to be replicated
├── src           # Code to run any relevant data acquisition/processing pipelines
├──── datasources        # Code to create each feature of the model
├──── utils              # Set up Blob storage + grid definition
├──── model_training.py  # Code to define the model and train it on historical data
├── main.py       # Code to predict impact data based on realtime forecasts
├── .github/workflows    # yaml file for setting up an Action for predicting impact on real-time events
|
├── .gitignore
├── README.md
└── requirements.txt

```

### **src** folder

This is the specific [README.md](https://github.com/OCHA-DAP/ds-aa-hti-hurricanes-impactmodel/blob/fede-implementation/src/README.md) file for this folder. There are also dedicated README.md files for each subfolder.

In **src/datasources**, we have multiple folders associated with each feature source that we will then use in the model. Here we collect and standardize:
- Historical hurricane tracks from IbTracks
- Associated accumulated rainfall for each storm from NASA PPS
- Population data from Meta Data for Good
- Topographical features from DWTKNS SRTM
- Vulnerability features (International Wealth Index) from GlobalDataLab
- Building footprint data from Google Open buildings

Additionally, we defined here the functions for collecting real-time weather data from ECMWF and GEFS.

In **src/utils**
- We define the functions for connecting to the blob storage (here we store historical and real-time events information). The user will need a *key* for this purpose (contact us).
- We define the grid cells for disaggregating every feature to the grid-level

In **src/model_training.py** we defined the 2-stage-grid-based-XGBoost model and train it on historical events.


### main.py script

This is the main script. Here:
- We collect real-time weather forecasts from **src/datasources/ECMWF** and **src/datasources/GEFS**
- Check if there's high-windspeed activity in the surroundings of Haiti
- If the ECMWF detects high windspeed (at least a Tropical Storm), an internal trigger activates
- If the trigger is activated, we use the pre-trained model to predict impact for each storm track the ECMWF detects
- The predictions for each storm track (storm_id, ensemble_member) is stored in a csv (aggregated at administrative 1 level)
- If the internal trigger doesn't activate, we don't predict anything.

### **.github/workflows**

Here we defined the YAML file for the automation of the running of the *main.py* script

### **exploration** folder

Here we study the historical impact data that we use as the target variable of our model. This data comes from the EMDAT database and consists of the total affected population (as defined by EMDAT, sum of: Total Deaths, No. Injured, No. Affected, No. Homeless) in cases of hurricanes in Haiti. We have impact information for 24 events between the years 2002-2021.

We also studied the return periods of these events based on the maximum recorded windspeed of each one.



## Development

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

The `markdownlint` hook will require
[Ruby](https://www.ruby-lang.org/en/documentation/installation/)
to be installed on your computer.

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```

It is also **strongly** recommended to use `jupytext`
to convert all Jupyter notebooks (`.ipynb`) to Markdown files (`.md`)
before committing them into version control. This will make for
cleaner diffs (and thus easier code reviews) and will ensure that cell outputs aren't
committed to the repo (which might be problematic if working with sensitive data).

---

[1] Towards a global impact-based forecasting model for tropical cyclones. Kooshki Forooshani, Mersedeh and van den Homberg, Marc and Kalimeri, Kyriaki and Kaltenbrunner, Andreas and Mejova, Yelena and Milano, Leonardo and Ndirangu, Pauline and Paolotti, Daniela and Teklesadik, Aklilu and Turner, Monica L. Natural Hazards and Earth System Sciences, 2024.


---
For recommendations, suggestions, concerns, etc. Contact me!
- Federico Moss (fedemoss@gmail.com)
