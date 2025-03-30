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