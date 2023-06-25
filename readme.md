## CO2e Estimation project :evergreen_tree:

**Project description:**
Main idea is to estimate the amount of CO2 captured by forest. 
For this purpose we use satellite data from Sentinel and GEDI to train Machine Learning model that calculates the CO2e collected by forest in an area of interest estimating Aboveground Biomass density.

Project install steps:
- clone repo
`git clone https://github.com/vashchuko/biomass_estimation.git`
- create virtual environment
`python -m venv env`
- activate virtual environment `source env/bin/activate` (for Linux and MacOS)
- install dependecies using pip
`pip install -r requirements.txt`
- pull needed data using dvc (it can take up to 20 minutes, please be patient, to see the progress we use verbose flag in here)
`dvc pull -v`
- reproduce training pipeline
`dvc repro`

[DVC documentation](https://dvc.org/doc/start/data-management/data-versioning)

Run app:
 - run `python app.py`
 - open http://127.0.0.1:5000/ in your browser and upload geoJSON from `data/nature_reserves/nature_reserves.geojson`
 - wait for results

In order to test endpoint - use subregions from data/nature_reserves/sub_regions

For more information reach us via slack