#!/usr/bin/env python3

# keras_train.py
# --------------------
# Use Keras to train a simple CNN to predict a discrete
# indicator of forage quality (inversely related to drought severity) from satellite
# images in 10 frequency bands. The ground truth label is the number of
# cows that a human expert standing at the center of the satellite image at ground level
# thinks the surrounding land could support (0, 1, 2, or 3+)

from drought_detection.data import load_dataset

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # for numpy calculations of composite bands
tf.get_logger().setLevel('ERROR') # only show errors, not warnings
tf.compat.v1.set_random_seed(89)

# Band key:
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

# adjustable settings/hyperparams
# these defaults can be edited here
MODEL_NAME = "models/efficientnetb0_all11bands_b64"
INPUT_BANDS = 'all'   # input bands to use for training (composite, rgb, all, or custom list)
NUM_BANDS = 11          # number of bands being used (all = 11, composite = 3, rgb = 3)
DATA_PATH = "data"
BATCH_SIZE = 64
EPOCHS = 50


# fixed variables (do not alter!)
# for categorical classification, there are 4 classes: 0, 1, 2, or 3+ cows
NUM_CLASSES = 4
# define class weights to account for uneven distribution of classes
# weights for training:         distribution of ground truth labels:
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
EFFICIENT_NET = 'b0'    # EfficientNet model
# performance optimization parameter
AUTOTUNE = tf.data.AUTOTUNE


# modelling
#----------------------------------
def build_efficientnet_from_scratch(num_bands, batch_size):
    # build model
    input_tensor = tf.keras.layers.Input(
        shape=(IMG_RESIZE, IMG_RESIZE, num_bands), name="image"
        )

    print("=========================build B0 model==============================")
    model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        input_tensor=input_tensor,
        weights=None,
        classes=NUM_CLASSES
        )


    # cyclical learning rate (for Adam optimizer)
    STEPS_PER_EPOCH = NUM_TRAIN // batch_size
    INIT_LR = 1e-5 # 0.00001
    MAX_LR = 1e-3 # 0.001
    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=INIT_LR,
        maximal_learning_rate=MAX_LR,
        scale_fn=lambda x: 1/(2.**(x-1)),
        step_size=2 * STEPS_PER_EPOCH
        )

    print("===================compile with cyclical lr==========================")

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    return model

def train_model(train_dataset, val_dataset, num_bands):
    # build model
    model = build_efficientnet_from_scratch(num_bands, BATCH_SIZE)

    print("===========================train model===============================")
    TRAIN_STEPS = NUM_TRAIN // BATCH_SIZE
    VAL_STEPS = NUM_VAL // BATCH_SIZE

    # train model
    model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        class_weight=CLASS_WEIGHTS,
        verbose=1
    )

    print("============================save model===============================")
    model.save(MODEL_NAME, overwrite = True)


if __name__ == "__main__":
    if INPUT_BANDS == 'all':
        NUM_BANDS = 11
    elif INPUT_BANDS == 'rgb' or INPUT_BANDS == 'composite':
        NUM_BANDS = 3
    else:
        NUM_BANDS = len(INPUT_BANDS)


    print('=====================================================================')
    print('=====================================================================')
    print(f'model and data parameters:\n\
        data path: {DATA_PATH}\n\
        model name: {MODEL_NAME}\n\
        efficient net version: {EFFICIENT_NET}\n\
        batch size: {BATCH_SIZE}\n\
        epochs: {EPOCHS}\n\
        input bands: {INPUT_BANDS}\n\
        number bands: {NUM_BANDS}\n'
        )
    print('=====================================================================')
    print('=====================================================================')


    print("============================load data================================")
    train_dataset, val_dataset = load_dataset(
        data_path=DATA_PATH,
        input_bands=[INPUT_BANDS],
        batch_size=BATCH_SIZE,
        train_shuffle_size=NUM_TRAIN
        )

    train_model(train_dataset, val_dataset, NUM_BANDS)
