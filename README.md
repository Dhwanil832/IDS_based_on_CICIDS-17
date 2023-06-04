# IDS based on CICIDS 2017 

This repository contains code for IDS based on the CICIDS 2017 dataset using machine learning techniques. The CICIDS 2017 dataset is a collection of network traffic data captured during different cyber attacks and normal network traffic. The goal of this project is to develop models that can classify network traffic instances as either normal or malicious.


## Dataset Description

The CICIDS 2017 dataset consists of several CSV files, each representing a different type of cyber attack or normal network traffic. The following files were used in this analysis:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Network traffic data during a DDoS attack in the afternoon on a Friday.
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: Network traffic data during a port scanning attack in the afternoon on a Friday.
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`: Network traffic data in the morning on a Friday.
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: Network traffic data during an infiltration attack in the afternoon on a Thursday.
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: Network traffic data during web-based attacks in the morning on a Thursday.

You can access the dataset from the following links:
-`https://www.kaggle.com/datasets/cicdataset/cicids2017`

## Code Overview

The code in this repository performs the following steps:

1. Data loading and preprocessing:
   - Load the dataset from the CSV files using `pd.read_csv`.
   - Concatenate the individual dataframes into a single dataframe.
   - Convert integer and float columns to lower precision data types.
   - Encode the categorical labels as numerical values.

2. Data balancing:
   - Perform undersampling on the majority class using `RandomUnderSampler`.
   - Perform oversampling on the minority class using `SMOTE`.

3. Model training and evaluation:
   - Split the data into training and testing sets using `train_test_split`.
   - Scale the numerical attributes using `StandardScaler`.
   - Train multiple classifiers, including Random Forest, Decision Tree, and Support Vector Machine.
   - Evaluate the models using cross-validation and calculate accuracy, confusion matrix, and classification report.

4. Ensemble modeling:
   - Create a Voting Classifier that combines the predictions of the individual models.
   - Train the Voting Classifier and evaluate its performance.


## Prerequisites

Make sure you have the following libraries installed before running the code:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`


## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Dhwanil832/IDS_based_on_CICIDS-17.git
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

3. Run the code:
- `Open the Jupyter Notebook or Python IDE of your choice.`
- `Open the cicids-pipeline-90-f1-score.ipynb notebook.`
- `Run the code cells sequentially to perform the data analysis.`


## Results

The analysis provides the following results for each model:

- `Cross-validation mean score`
- `Model accuracy`
- `Confusion matrix`
- `Classification report`
- `Additionally, an ensemble model (Voting Classifier) is trained and evaluated for improved performance.`


## Conclusion

The code in this repository demonstrates the process of a basic IDS based on the CICIDS 2017 dataset using various machine learning models. By training and evaluating these models, we can effectively classify network traffic instances as normal or malicious