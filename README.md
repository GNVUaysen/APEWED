# APEWED – Air Pollution Early Warning of Emergency Demand

This repository contains the code and aggregated datasets supporting the study
"AI-based early warning model for respiratory emergency demand driven by air pollution and meteorological conditions", submitted to *Urban Climate*.

## Contents
- Python scripts for data preprocessing, statistical analysis, and machine learning
- Aggregated and anonymized datasets
- Reproducible analysis pipeline

## Data availability
Air quality and meteorological data are publicly available from the Chilean National Air Quality Information System (SINCA; https://sinca.mma.gob.cl/index.php/redes). All emergency healthcare data are aggregated were extracted from Chilean Department of Health Statistics and Information (DEIS; deis.minsal.cl), Ministry of Health. 

## Code availability
All analysis scripts were developed in Python and are provided in this repository.

## Requirements
- Python 3.x

Main Python packages used in this project:
- numpy
- pandas
- scipy
- statsmodels
- scikit-learn
- joblib
- openpyxl
- xgboost
- catboost
- lightgbm (optional)

## Usage
python3 predict_48h.py --input example_predict.csv --model final_model_operational_48h.joblib

## Disclaimer
This repository does not contain identifiable patient data.

