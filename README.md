# ProEnFo

## Introduction

This is the code related to the paper 
"Benchmarks and Custom Package for Electrical Load Forecasting"(https://openreview.net/forum?id=O61RXF9dvD&invitationId=NeurIPS.cc/2023/Track/Datasets_and_Benchmarks/Submission433/-/Supplementary_Material&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FTrack%2FDatasets_and_Benchmarks%2FAuthors%23author-tasks)) submitted to Neurips 2023 Datasets and Benchmarks Track. 
This repository mainly aims at implementing routines for probabilistic energy forecasting. However, we also provide the implementation of relevant point forecasting models.
The datasets and their forecasting results in this archive can be found at https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3009646_connect_hku_hk/Euy4Rv8DsM1Cu1hJ85yHL18BNsDNbS5XiaVoCvl-l-07tQ?e=OFLF3A. 
To reproduce the results in our archive, users can refer to the process in the main.py file. By selecting different Feature engineering methods and preprocessing, post-processing, and training models, users can easily construct different prediction models.

## Dataset
	Dataset	No. of series	Length	Resolution	Load type	External variables
1	Covid19	1	31912	hourly	aggregated	airTemperature,
Humidity, etc
2	GEF12	20	39414	hourly	aggregated	airTemperature
3	GEF14	1	17520	hourly	aggregated	airTemperature
4	GEF17	8	17544	hourly	aggregated	airTemperature
5	PDB	1	17520	hourly	aggregated	airTemperature
6	Spanish	1	35064	hourly	aggregated	airTemperature,
seaLvlPressure, etc
7	Hog	24	17544	hourly	building	airTemperature,
wind speed, etc.
8	Bull	41	17544	hourly	building	airTemperature,
wind speed, etc.
9	Cockatoo	1	17544	hourly	building	airTemperature,
wind speed, etc.
10	ELF	1	21792	hourly	aggregated	No
11	UCI	321	26304	hourly	building	No


## Prerequisites
- Python 
- Conda

### Create a virtual environment
This is only needed when used the first time on the machine.

```bash
conda env create --file proenfo_env.yml
```

### Activate and deactivate enviroment
```bash
conda activate proenfo_env
conda deactivate
```

### Update your local environment

If there's a new package in the `proenfo_env.yml` file you have to update the packages in your local env

```bash
conda env update -f proenfo_env.yml
```

### Export your local environment

Export your environment for other users

```bash
conda env export > proenfo_env.yml 
```

### Recreate environment in connection with Pip
```bash
conda env remove --name proenfo_env
conda env create --file proenfo_env.yml
```

### Initial packages include
  - python=3.9.13
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - plotly
  - statsmodels
  - xlrd
  - jupyterlab
  - nodejs
  - mypy
  - pytorch
