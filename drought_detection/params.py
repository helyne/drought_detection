
### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account


### GCP Project - - - - - - - - - - - - - - - - - - - - - -

GCP_PROJECT_ID = 'drought-detection'


### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-batch913-drought_detection'


##### GCP Data  - - - - - - - - - - - - - - - - - - - - - - - -

# butcket data file location
BUCKET_TRAIN_DATA_PATH = 'data/train'
BUCKET_VAL_DATA_PATH = 'data/val'

# # GS bucket location path
# GS_BUCKET_PATH_LINK = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"


### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# GCP_AI_PLATFORM =

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[DEU] [berlin] [helyne] drought_baseline + v1"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

# AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"



##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'drought_baseline'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'
