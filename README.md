<!-- # DP_Thesis

## How to Run this project

1. Download defectors.zip from this link https://zenodo.org/records/7708984 and extract it into /data/raw folder.

2. Install the python dependencies listed in ./Pipfile or in requirements.txt

3. Run the Python script in ./src_code/etl/ETL.py to extract the feature space. Run with --help argument to get help on how to use the file.

4. Run the script in ./src_code/ml_pipeline/scripts/engineer.py to enrich feature space with engineered metrics. Run with --help argument to get help on how to use the file.

5. Run the script in ./src_code/ml_pipeline/scripts/EDA.py to perform EDA on engineered feature set. Run with --help argument to get help on how to use the file.

6. If you want to performed EDA on transformed, training-ready data, run the script in ./src_code/ml_pipeline/scripts/preprocess.py to apply preprocessing & transformations on the engineered dataset. Now rerun the EDA script with different params. Run with --help argument to get help on how to use the files.

7. To perform hyperparameter tuning, run the script in ./src_code/ml_pipeline/scripts/tune.py. Run with --help argument to get help on how to use the file.

8. To train any of the supported models, run the script in ./src_code/ml_pipeline/scripts/train.py. Use --load-tuned to reconfigure model with optimal params if needed. Run with --help argument to get help on how to use the file.

9. To perform combined evaluation of the trained models, run the script in ./src_code/ml_pipeline/scripts/evaluate.py. Use --models to specify a subset of supported models. Run with --help argument to get help on how to use the file. -->

# DP_Thesis

## Overview
This repository contains the implementation of a machine learning pipeline for Just-In-Time (JIT) defect prediction at the commit level. The project covers the full workflow, including data extraction (ETL), feature engineering, exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Setup Instructions

### 1. Dataset Preparation
Download the Defectors dataset:
https://zenodo.org/records/7708984

Extract the contents of `defectors.zip` into:
/data/raw

---

### 2. Install Dependencies
Install required Python packages using either:

pip install -r requirements.txt

or (if using Pipenv):

pipenv install

---

## Pipeline Execution

The project is structured as a modular pipeline. Each stage is executed via dedicated scripts:

### 3. Feature Extraction (ETL)
python ./src_code/etl/ETL.py --help

---

### 4. Feature Engineering
python ./src_code/ml_pipeline/scripts/engineer.py --help

---

### 5. Exploratory Data Analysis (EDA)
python ./src_code/ml_pipeline/scripts/EDA.py --help

---

### 6. Preprocessing (Optional for Transformed EDA)
python ./src_code/ml_pipeline/scripts/preprocess.py --help

You can then rerun the EDA script to analyze transformed data.

---

### 7. Hyperparameter Tuning
python ./src_code/ml_pipeline/scripts/tune.py --help

---

### 8. Model Training
python ./src_code/ml_pipeline/scripts/train.py --help

To use tuned hyperparameters:
python train.py --load-tuned

---

### 9. Model Evaluation
python ./src_code/ml_pipeline/scripts/evaluate.py --help

To evaluate specific models:
python evaluate.py --models RF XGB NN

---

## Notes
- All scripts support the `--help` flag for detailed usage instructions.
- Outputs such as models, logs, and reports are versioned for reproducibility.
- The pipeline is designed to be modular, enabling independent execution of each stage.

---

## Structure Summary
- etl/ – raw data extraction
- ml_pipeline/ – EDA, preprocessing, training, evaluation
- models/ – stored models and artifacts
- reports/ – EDA and evaluation outputs
- notebooks/ – exploratory and analysis notebooks

---

## License
Specified in LICENSE file.
