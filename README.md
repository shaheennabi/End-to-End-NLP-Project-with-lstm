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