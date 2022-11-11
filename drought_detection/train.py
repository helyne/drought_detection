#!/usr/bin/env python3

# --------------------
# Use Keras to train a simple CNN to predict a discrete
# indicator of forage quality (inversely related to drought severity) from satellite
# images in 10 frequency bands. The ground truth label is the number of
# cows that a human expert standing at the center of the satellite image at ground level
# thinks the surrounding land could support (0, 1, 2, or 3+)

import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # for numpy calculations of composite bands
tf.get_logger().setLevel('ERROR') # only show errors, not warnings
tf.compat.v1.set_random_seed(89)


from drought_detection.data import load_dataset, image_augmentation
from drought_detection.params import MODEL_NAME, INPUT_BANDS, \
    DATA_PATH, BATCH_SIZE, EPOCHS, NUM_CLASSES, CLASS_WEIGHTS, \
    NUM_TRAIN, NUM_VAL, IMG_RESIZE

# performance optimization parameter
AUTOTUNE = tf.data.AUTOTUNE


# modelling
#----------------------------------

def build_efficientnet_from_scratch(num_bands, batch_size):
    # build model
    # input_tensor = tf.keras.layers.Input(
    #     shape=(IMG_RESIZE, IMG_RESIZE, num_bands), name="image"
    #     )
    inputs = tf.keras.layers.Input(shape=(IMG_RESIZE, IMG_RESIZE, num_bands), name="image")
    # x = image_augmentation(inputs)
    x = inputs

    print("=========================build B0 model==============================")
    # model = tf.keras.applications.EfficientNetB0(
    #     include_top=True,
    #     input_tensor=input_tensor,
    #     weights=None,  # type: ignore
    #     classes=NUM_CLASSES
    #     )

    outputs = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None, # type: ignore
        classes=NUM_CLASSES)(x)

    model = tf.keras.Model(inputs, outputs)


    # cyclical learning rate (for Adam optimizer)
    STEPS_PER_EPOCH = NUM_TRAIN // batch_size
    INIT_LR = 1e-5 # 0.00001
    MAX_LR = 1e-3 # 0.001

    def scale_fn(x):
        return 1. ** x

    lr = CyclicalLearningRate(initial_learning_rate=1e-5,
                              maximal_learning_rate=1e-3,
                              step_size=2*STEPS_PER_EPOCH,
                              scale_fn=scale_fn)
    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr) # type: ignore

    print("============compile with Adam optimizer & cyclical lr================")

    # compile model
    model.compile(
        optimizer=optimizer,
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
        verbose=1  # type: ignore
        )

    print("============================save model===============================")
    model.save(MODEL_NAME, overwrite = True)


if __name__ == "__main__":
    if INPUT_BANDS == ['all']:
        NUM_BANDS = 11
    elif INPUT_BANDS == ['rgb'] or INPUT_BANDS == ['composite']:
        NUM_BANDS = 3
    else:
        NUM_BANDS = len(INPUT_BANDS)


    print('=====================================================================')
    print('=====================================================================')
    print(f'model and data parameters:\n\
        data path: {DATA_PATH}\n\
        model path: {MODEL_NAME}\n\
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
        input_bands=INPUT_BANDS,
        batch_size=BATCH_SIZE,
        train_shuffle_size=NUM_TRAIN
        )

    train_model(train_dataset, val_dataset, NUM_BANDS)
