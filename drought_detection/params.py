
# ============================ BAND REFERENCE KEY ==============================

# B1  30 meters   0.43 - 0.45 µm  Coastal aerosol
# B2  30 meters   0.45 - 0.51 µm  Blue
# B3  30 meters   0.53 - 0.59 µm  Green
# B4  30 meters   0.64 - 0.67 µm  Red
# B5  30 meters   0.85 - 0.88 µm  Near infrared
# B6  30 meters   1.57 - 1.65 µm  Shortwave infrared 1
# B7  30 meters   2.11 - 2.29 µm  Shortwave infrared 2
# B8  15 meters   0.52 - 0.90 µm  Band 8 Panchromatic
# B9  15 meters   1.36 - 1.38 µm  Cirrus
# B10 30 meters   10.60 - 11.19 µm Thermal infrared 1, resampled from 100m to 30m
# B11 30 meters   11.50 - 12.51 µm Thermal infrared 2, resampled from 100m to 30m



# =========================== DATA & MODEL LOCATIONS ===========================
# ================================ DO NOT ALTER ================================


### GCP Storage - - - - - - - - - - - - - - - - - - - - - -
GCP_PROJECT_ID = 'drought-detection'
GCP_BUCKET_NAME = 'wagon-data-batch913-drought_detection'
GCP_DATA_PATH = f'gs://wagon-data-batch913-drought_detection/data'
GCP_MODELS_PATH = f'gs://wagon-data-batch913-drought_detection/models'

### Local Storage - - - - - - - - - - - - - - - - - - - - - -
LOCAL_DATA_PATH = 'data'
LOCAL_MODELS_PATH = 'models'


# ========================= STABLE TRAINING PARAMETERS =========================
# ================================ DO NOT ALTER ================================

# for categorical classification, there are 4 classes: 0, 1, 2, or 3+ cows
NUM_CLASSES = 4

# class weights to account for uneven distribution of classes
# training weights:             distribution of ground truth labels:
CLASS_WEIGHTS = {0: 1.0,        # 0: ~60%
                1: 4.0,         # 1: ~15%
                2: 4.0,         # 2: ~15%
                3: 6.0}         # 3: ~10%

# fixed image counts from TFRecords
NUM_TRAIN = 85056       # 86317 before filtering for null images
NUM_VAL = 10626         # 10778 before filtering for null images

# default image size resolution dimension (65 x 65 pixel square)
IMG_DIM = 65

# resizing resolution dimension is determined by EfficientNet model choice
IMG_RESIZE = 224        # EfficientNetB0 = 224

# efficientnet model version
EFFICIENT_NET = 'b0'    # EfficientNet model



# ==============================================================================
# ======================= ADJUSTABLE TRAINING PARAMETERS =======================
# ==============================================================================

# list of input bands to use for training
INPUT_BANDS = ['rgb']     # can be one of:
                            # ['all']
                            # ['composite']
                            # ['rgb']
                            # or custom list (eg. ['B3', 'B5', 'B7', 'B2'])

# batch size of samples
BATCH_SIZE = 64

# number of training epochs
EPOCHS = 50

# model save path
MODEL_NAME = f'{LOCAL_MODELS_PATH}/rgbbands_b64_e50'

# directory to train and val data
DATA_PATH = LOCAL_DATA_PATH
