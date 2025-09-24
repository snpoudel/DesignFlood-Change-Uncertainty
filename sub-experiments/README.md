# Sensitivity Analysis: Sub-Experiments

Several sub-experiments were conducted to assess the sensitivity of the main experiment to different common model calibration techniques. These included six sub-experiments:

## 1. Different Climate Change Scenarios
The main experiment was conducted under four different climate change scenarios. The scripts are similar to the main experiment, but with different meteorological forcing files generated using a stochastic weather generator. Also the main experiment used 1040-years of data, so the analysis was repeated using only 50-years of data to see the effect of data length in extreme value analysis.

## 2. Alternative Fitting Methods
The main experiment was repeated using different extreme value distribution models and fitting methods. The scripts for estimating flood changes with various methods, along with the corresponding plots, are provided in the `Exp-alternative-fitting-methods` folder.

## 3. Varying Number of Iterations and Training Data Length for Process Models
The main experiment was repeated with different numbers of iterations and varying training data lengths for process models. The scripts is similar to the main experiment, with adjustments made only to the iteration number of optimizing algorithm and training data length.

## 4. Limiting Training Data Length for Deep Learning Models
The main experiment was repeated reduced training data lengths for deep learning models (both standard and hybrid). The scripts is same as the main experiment, with only change applied to the training data length used for training.

## 5. Handling Equifinality with Multi-Variable Optimization
To address equifinality, the main experiment was repeated using multi-variable optimization approach. Optimization was performed to minimize the residuals of both streamflow and evapotranspiration simultaneously. The related scripts for models, performance evaluation, flood change analysis, and plotting are provided in the `Exp-equifinality-multivariable` folder.

## 6. Handling Equifinality with Multi-Objective Optimization
To address equifinality, the main experiment was repeated using multi-objective optimization approach. Here, optimization was carried out to simultaneously minimize different components of streamflow, including low flow, high flow, and total flow. The corresponding scripts for models, performance evaluation, flood change analysis, and plotting are provided in the `Exp-equifinality-multiobjective` folder.

## 7. Sobol Sensitivity Analysis
To quantitatively attribute how much the error variability in flood change projections is due to different sources of uncertainty (hydrological model types, precipitation error, and climate change scenarios), a Sobol sensitivity analysis was conducted. The scripts for performing the Sobol sensitivity analysis and generating the related plots are provided in the `Exp-sobol-sensitivity-analysis` folder.

## 8. Convergent Cross Mapping
To explore the impact of equifinality on the flood change projection using a causal inference approach, Convergent Cross Mapping (CCM) technique was used. The scripts for performing CCM analysis and generating the related plots are provided in the `Exp-convergent-cross-mapping` folder.

## 9. GR4J as Truth Model
The entire experimental setup was repeated using GR4J as the truth model instead the original HBV model. The script for developing the GR4J truth models, and developing competing models based on the GR4J truth models, and the analysis for comparision of model performance and flood change projections are provided in the `Exp-GR4J-truth-model` folder.