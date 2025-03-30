# Hydrological Change Uncertainty
## This repository contains the scripts and data used for the study "Title of the Paper" published in *Name of Journal*. 
## The study investigates the impact of input, model, and parameter uncertainty on hydrological change projections over 30 basin in Massachussetts, US. A set of hydrological models including process-based models, deep learning model (LSTM), and hybrid models (LSTM + process-based models) are used to estimate the hydrological changes under different climate scenarios. The link to the paper is provided here: [Link to the paper](https:link).

## The repository is organized into the following sections:
- **data**: The study used a stochastic weather generator to create 1040-years of synthetic weather data for 30 basins in Massachusetts, US. The size of total data is around 1000 GB, which is too large to provide in this repository. The acquistion and processing of the data is described in the paper. The final processed data that came out of the analysis is provided here which can be used to reproduce the figures in the paper.
- **figures**: Contains the figures along with corresponding script used to generate the figures.
- **scripts**: Contains the scripts used for the analysis.

## The python scripts are created in a sequential manner in terms of data pre-processing, model training and prediction, and model evaluation, analysis, and visualization. The scripts are organized into the following sections:

- **pre-processing**: Scripts used to pre-process the data, the pre-processed inputs are fed into the hydrological models. Here are the list of pre-processing scrips with a brief description:
    - `1.process_swg.py`: This script is used to process the output file format of the stochastic weather generator to a format that is easier to work with. This scripts outputs csv files of the weather data (temperature, precipitatoin) for each gauging station in the study area for 1040-years and for a baseline and four future climate scenarios.
    - '2.1true_precip.py': This script calculates the areal average true precipitation by inverse distance weighting interpolation of the precipitation data. The true precip for a basin represents precip estimated using all gauging stations available for the basin.
    - '2.2future_true_precip.py': This script calculates the areal average true precipitation for the future climate scenarios. The true precip for a basin represents precip estimated using all gauging stations available for the basin.
    - '2.3noisy_precip_hist_future.py': This script calculates the areal average noisy precipitation for the historical and future climate scenarios. The noisy precip for a basin represents precip estimated by randomly sampling the precipitation gauges available for the basin.
    - '2.4temperature_hist_future.py': This script calculates mean temperature for the historical and future climate scenarios.
    - '2.5regional_lstm_precip_bucket_histogram.py;" This script calculates the error in terms of RMSE of noisy precipitaiton by comparing with the true precipitation. Then categorizes the precipitation data into different buckets of error.
    - '2.6prepare_regional_lstm_train_dataset.py': This script combines the catchment statics and dynamics to create a dataset that is used to train the regional LSTM model. The regional LSTM model is prepared for each precipitation error bucket.
    - '2.7prepare_regional_lstm_prediction_dataset.py': This script prepares the dataset that the regional trained with the training dataset will use to make predictions. The dataset is prepared for each precipitation error bucket.
    - '2.8prepare_hymod_lstm_dataset.py': This script prepares both the training and prediction dataset for the hybrid model (FULL-HYMOD + LSTM). The dataset is prepared for each precipitation error bucket.
    - '2.9prepare_simp_hymod_lstm_dataset.py': This script prepares both the training and prediction dataset for the hybrid model (HYMOD + LSTM). The dataset is prepared for each precipitation error bucket.

- **model training and prediction**: Scripts used to train the hydrological models and make predictions. Here are the list of model training and prediction scrips with a brief description:
    - '3.1hbv_true.py': This script runs the truth model (hbv_model.py) to generate streamflow simulation using true precipitation data for all study basins.
    - '3.2hbv_noisy.py': This script runs the truth model (hbv_model.py) to generate streamflow simulation using noisy precipitation data for all study basins.
    - '3.3rehbv.py': This script recalibrates the truth model (hbv_model.py) using the noisy precipitation data to generate streamflow simulation for all study basins.
    - '3.4hymod.py': This script runs the hymod model (hymod_model.py) to generate streamflow simulation using both true and noisy precipitation data for all study basins.
    - '3.5simp_hymod.py': This script runs the simplified hymod model (simp_hymod_model.py) to generate streamflow simulation using both true and noisy precipitation data for all study basins.
    - '3.6regionalLSTM.py': This script trains the regional LSTM model and make prediction for each precipitation error bucket. 
    - '3.7hymod_lstm.py': This script trains the hybrid model (FULL-HYMOD + LSTM) and make prediction for each precipitation error bucket.
    - '3.7hymod_lstm_summed_output.py': This script simply sums the output of the hybrid model (FULL-HYMOD + LSTM) to the output of the FULL-HYMOD model to get the final output of the hybrid model.
    - '3.8simp_hymod_lstm.py': This script trains the hybrid model (HYMOD + LSTM) and make prediction for each precipitation error bucket.
    - '3.8simp_hymod_lstm_summed_output.py': This script simply sums the output of the hybrid model (HYMOD + LSTM) to the output of the HYMOD model to get the final output of the hybrid model.
    - '3.9lstm_tuning.py': This script runs 4-fold cross-validation of the regional LSTM model and provides the best hyperparameters values which are used in the regional LSTM model training.
    - '3.10residual_lstm_tuning.py': This script runs 4-fold cross-validation of the residual LSTM model and provides the best hyperparameters values which are used in the residual LSTM model (hybrid) training.
    - '3.11plot_lstm_tuning.py': This script plots the results of the lstm tuning and residual lstm tuning scripts.