# Uncertainties in Hydrological Change Projections under Climate Change

🚧 **Repository Under Construction** 🚧

Welcome to the repository for **"Title of the Paper"**, published in *Name of Journal*.
The study investigates the impact of input, model, and parameter uncertainty on hydrological change projections across 30 basin in Massachussetts, US. A set of hydrological models including process-based models, deep learning model (LSTM), and hybrid models (LSTM + process-based models) are used to estimate the hydrological changes under different climate scenarios. 

📄 Read the full paper here: [Link to the paper](https://link)

Citation:

```bibtex
@article{your_citation_key,
  title={Title of the Paper},
  author={Author1, Author2, Author3},
  journal={Name of Journal},
  year={2023},
  volume={X},
  number={Y},
  pages={Z-ZZ},
  publisher={Publisher}
}
```

---

## 📂 Repository Structure

This repository is organized into the following sections:

### 1️⃣ **data** 
- The study used a stochastic weather generator to create 1040-years of synthetic weather data for 30 basins in Massachusetts, US. The size of total data is around 1000 GB, which is too large to provide in this repository. The acquistion and processing of the data is described in the paper. The final processed data that came out of the analysis is provided here which can be used to reproduce the figures in the paper.

### 2️⃣ **figures** 
- Contains the figures along with corresponding script used to generate the figures.

### 3️⃣ **scripts** 
- Contains scripts for data processing, model training, prediction, and analysis.

---

## **scripts Overview**
The scripts are structured **sequentially**, covering:

1. **Pre-processing**
2. **Model Training & Prediction**
3. **Model Evaluation, Analysis & Visualization**

---

### 🔹 **1. Pre-processing Scripts**
Scripts for preparing the dataset before feeding into hydrological models.

| Script | Description |
|--------|------------|
| `1.process_swg.py` | Converts stochastic weather generator output into a usable CSV format. |
| `2.1true_precip.py` | Computes areal-averaged true precipitation using **Inverse Distance Weighting (IDW)**. |
| `2.2future_true_precip.py` | Computes true precipitation for future climate scenarios. |
| `2.3noisy_precip_hist_future.py` | Computes areal-averaged **noisy precipitation** for historical and future scenarios. |
| `2.4temperature_hist_future.py` | Computes mean temperature for historical and future scenarios. |
| `2.5regional_lstm_precip_bucket_histogram.py` | Categorizes precipitation data into **error buckets** using RMSE. |
| `2.6prepare_regional_lstm_train_dataset.py` | Prepares dataset for training the **Regional LSTM Model** per error bucket. |
| `2.7prepare_regional_lstm_prediction_dataset.py` | Prepares dataset for **regional LSTM model predictions**. |
| `2.8prepare_hymod_lstm_dataset.py` | Prepares dataset for **FULL-HYMOD + LSTM hybrid model**. |
| `2.9prepare_simp_hymod_lstm_dataset.py` | Prepares dataset for **HYMOD + LSTM hybrid model**. |

---

### 🔹 **2. Model Training & Prediction**
Scripts for training hydrological models and making predictions.

| Script | Description |
|--------|------------|
| `3.1hbv_true.py` | Runs **HBV model** (`hbv_model.py`) using true precipitation data. |
| `3.2hbv_noisy.py` | Runs **HBV model** (`hbv_model.py`) using noisy precipitation data. |
| `3.3rehbv.py` | Recalibrates **HBV model** (`hbv_model.py`) using noisy precipitation data. |
| `3.4hymod.py` | Runs **HYMOD model** (`hymod_model.py`) using true and noisy precipitation. |
| `3.5simp_hymod.py` | Runs **simplified HYMOD model** (`simp_hymod_model.py`) using true and noisy precipitation. |
| `3.6regionalLSTM.py` | Trains and predicts using the **Regional LSTM model** per error bucket. |
| `3.7hymod_lstm.py` | Trains and predicts using the **FULL-HYMOD + LSTM hybrid model**. |
| `3.7hymod_lstm_summed_output.py` | Computes final **FULL-HYMOD + LSTM** output by summing model outputs. |
| `3.8simp_hymod_lstm.py` | Trains and predicts using the **HYMOD + LSTM hybrid model**. |
| `3.8simp_hymod_lstm_summed_output.py` | Computes final **HYMOD + LSTM** output by summing model outputs. |
| `3.9lstm_tuning.py` | Runs **4-fold cross-validation** for LSTM model hyperparameter tuning. |
| `3.10residual_lstm_tuning.py` | Runs **4-fold cross-validation** for residual LSTM model tuning. |
| `3.11plot_lstm_tuning.py` | Visualizes LSTM tuning results. |

---

### 🔹 **3. Model Evaluation, Analysis & Visualization**
Scripts for model assessment, result analysis, and visualization.

#### **📌 Model Parameters Visualization**
| Script | Description |
|--------|------------|
| `4.1plot_HBVparams.py` | Plots **HBV model parameters** for a given basin. |
| `4.2plot_HYMOD_params.py` | Plots **HYMOD model parameters**. |
| `4.3plot_SIMP-HYMOD_params.py` | Plots **simplified HYMOD model parameters**. |

#### **📌 Simulated Streamflow Visualization**
| Script | Description |
|--------|------------|
| `5.1plot_precip_streamflow_timeseries.py` | Plots **precipitation & streamflow time series**. |
| `5.2plot_returnperiods.py` | Plots **return periods of streamflow**. |

#### **📌 Model Summary Performance Metrics**
| Script | Description |
|--------|------------|
| `6.1summary_metrics_diagnostics.py` | Computes **summary metrics** for model performance. |
| `6.2plot_summary_metrics.py` | Visualizes **summary metrics**. |

#### **📌 Flood Change Analysis**
| Script | Description |
|--------|------------|
| `7.1estimate_flood_change.py` | Estimates **flood change** (e.g., 25-year return period). |
| `7.2plot_flood_change.py` | Visualizes flood change for all basins. |
| `7.3estimate_pooled_flood_change.py` | Computes **pooled flood change** estimates. |
| `7.4plot_pooled_flood_change.py` | Visualizes **pooled flood change** results. |

