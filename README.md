# Hydrological Change Uncertainty
## This repository contains the scripts and data used for the study "Title of the Paper" published in *Name of Journal*. 
## The study investigates the impact of input, model, and parameter uncertainty on hydrological change projections over 30 basin in Massachussetts, US. A set of hydrological models including process-based models, deep learning model (LSTM), and hybrid models (LSTM + process-based models) are used to estimate the hydrological changes under different climate scenarios. The link to the paper is provided here: [Link to the paper](https:link).

## The repository is organized into the following sections:
- **data**: The study used a stochastic weather generator to create 1040-years of synthetic weather data for 30 basins in Massachusetts, US. The size of total data is around 1000 GB, which is too large to provide in this repository. The acquistion and processing of the data is described in the paper. The final processed data that came out of the analysis is provided here which can be used to reproduce the figures in the paper.
- **figures**: Contains the figures along with corresponding script used to generate the figures.
- **scripts**: Contains the scripts used for the analysis.

## The python scripts can be categorized into three categories:
- **pre-processing**: Scripts used to pre-process the data, the pre-processed inputs are fed into the hydrological models. Here are the list of pre-processing scrips with a brief description:
    - `1..py`: This script is used to pre-process the data. It takes the raw data as input and generates the pre-processed data.
