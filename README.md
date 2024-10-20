# End-to-End-NLP-Project-with-RNN

This project aims to classify hate speech using a deep learning model. The dataset for this project was sourced from Kaggle and underwent significant preprocessing to ensure a balanced and clean training set.

### Project Overview
In this project, I tackled the challenge of detecting hate speech using an LSTM-based model. The dataset contained inherent imbalance, so I combined two separate datasets into one during preprocessing to ensure more robust model performance.

### Key Features:
* Model Architecture: LSTM (Long Short-Term Memory) neural network for text classification.
* Embedding Layer: Used Keras' embedding layer to handle word representations.
* Model Size: The model consists of approximately 5,080,501 total parameters.
* Data Preprocessing:
Combined two datasets to address imbalance issues.
Applied text preprocessing steps such as tokenization, lowercasing, and removal of stopwords.



# Project Tree Structure
``` bash

.
├── END-TO-END-NLP-PROJECT-WITH-LSTM
├── .circleci/
│   └── config.yml
├── artifact/
│   ├── 10_05_2024_03_23_14 (or time Stamp)/
│   │   ├── DataIngestionArtifacts/
│   │   │   ├── dataset.zip
│   │   │   ├── imbalanced_data.csv
│   │   │   └── raw_data.csv
│   │   ├── DataValidationArtifacts/
│   │   │   └── status.txt
│   │   ├── DataTransformationArtifacts/
│   │   │   └── final.csv
│   │   ├── ModelTrainerArtifacts/
│   │   │   ├── model.h5
│   │   │   ├── x_test.csv
│   │   │   ├── x_train.csv
│   │   │   └── y_test.csv        
│   │   └── ModelEvaluationArtifacts    /
│   │       └── best_model/
│   │           └── model.h5
│   └── PredicModel /
│       └── model.h5  
├── data/
│   └── dataset.zip
├── Hate or src/
│   ├── components/
│   │   ├── __pychache__/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   └── model_trainer.py
│   ├── configuration/
│   │   ├── __pycache__
│   │   ├── __init__.py
│   │   └── s3_syncer.py
│   ├── constants/
│   │   ├── __pycache__/
│   │   └── __init__.py
│   ├── entity/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/
│   │   ├── __pycache__/
│   │   └── __init__.py
│   ├── logger/
│   │   ├── __pycache__/
│   │   └── __init__py
│   ├── ml/
│   │   ├── __init__.py
│   │   └── model.py
│   └── pipeline/
│       ├── __pycache__/
│       ├── __init__.py
│       ├── training_pipeline.py
│       └── prediction_pipeline.py
├── logs/
│   └── 10_05_2024_03_23_14.log/
│       └── 10_05_2024_03_23_14.log
├── Notebook/
│   └── Hate_speech_experiment.ipynb
├── s3_downloads/
│   └── dataset.zip
├── app.py
├── circleci_setup_template.sh
├── Dockerfile
├── README.md
├── requirements.txt
├── setup.py
└── template.py

```



## How to run?
``` bash
conda create -n hate python=3.8 -y
```

``` bash
conda activate hate
```

``` bash
pip install -r requirements.txt
```

### Export the environment variable(gitbash)

``` bash
export AWS_ACCESS_KEY_ID="your access key"
```
``` bash
export AWS_SECRET_ACCESS_KEY="your secret access key"
```
``` bash
export AWS_DEFAULT_REGION="e.g, us-east-1"  
```




# Workflow
After creating project template
 * Update constants 
 * Update Entiry modules
 * Update respective component
 * Update pipeline
 


## Deployment

1. Setting up circleCI
2. Switch on self hosted runner
3. Create Project
4. Configure EC2
5. config.yml
6. env variables



## Code Training & Prediction pipeline working properly, updating my circleCI CICD deployment in future.