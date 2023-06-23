Above Ground Biomass Estimation project.

Project description:
Main idea is to estimate the amount of CO2 captured by forest. 
For this purpose we use satellite data from Sentinel and GEDI to train ML model that calculates the CO2e collected by forest in an area of interest. 

Project install steps:
- Clone repo
git clone https://github.com/vashchuko/biomass_estimation.git
- create virtual environment
python -m venv env
- activate virtual environment
- install dependecies using pip
pip install -r requirements.txt
- pull needed data using dvc (it can take up to 20 minutes, please be patient, to see the progress we use verbose flag in here)
dvc pull -v 
- reproduce training pipeline
dvc repro

For more information reach me via slack
DVC docs https://dvc.org/doc/start/data-management/data-versioning