# Reproducibility Guidelines
### This README provides instructions on how to reproduce the results presented in the paper using the CMP-CV framework (*CoMParison workflow for Cross Validation* of deep learning models) or by running individual models.

## Generating Predictions
### To obtain the results mentioned in the paper, you can either use the CMP-CV framework or run the individual models to generate predictions for the test set. If you conduct multiple training runs, please follow these steps:

### Save the results for each model in separate folders using the format: model_res_path/{i}, where i represents the i-th training run.
### Within each folder, save the test set predictions in CSV files named test_predictions.csv.

## CMP-CV Framework

### CMP-CV is a component of the Supervisor framework which was developed under CANDLE Exascale Computing Program, and the development can be found at https://github.com/ECP-CANDLE/Supervisor/tree/develop/workflows/cmp-cv. Please note that this repository may provide information about the authors of this work.

## Analysis Scripts

### To perform various analyses and generate plots, use the following scripts:

#### 1. `src/1.domain_errors.py` : This script computes and stores the domain errors of the molecular descriptors.
#### Run the script to calculate the domain errors.
#### The results will be saved for further analysis.

#### 2. `src/2.user_domain_errors.py` : This script generates domain error plots for user-specified drug descriptors.
#### Modify the script to specify the desired drug descriptors.
#### Run the script to create the domain error plots.
#### The generated plots will provide insights into the domain errors of the selected descriptors.

####  3. `src/3.model_ranks.py`  : This script obtains the model rankings in various molecular descriptor domains.
#### Run the script to calculate the model rankings.
#### The rankings will be displayed or saved, depending on the script's configuration.
#### Please ensure that you have the necessary dependencies and data files available before running these scripts.


#### 4. To run the web app use, `streamlit run web_app/app.py`
