import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa

def load_dataset():
    print("======================================starting======================================")

    # load the whole dataset, for data info
    all_ds   = tfds.load("eurosat", with_info=True)

    # load training, testing & validation sets, splitting by 60%, 20% and 20% respectively
    train_ds = tfds.load("eurosat", split="train[:60%]")
    test_ds  = tfds.load("eurosat", split="train[60%:80%]")
    valid_ds = tfds.load("eurosat", split="train[80%:]")


    # the class names
    class_names = all_ds[1].features["label"].names
    # total number of classes (10)
    num_classes = len(class_names)
    num_examples = all_ds[1].splits["train"].num_examples

    return train_ds, test_ds, valid_ds, num_examples, num_classes


def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
    print("===========================starting preprocessing========================")
    num_classes = 10
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
    # ds = ds.map(lambda d: (d["sentinel2"], tf.one_hot(d["label"], num_classes)))
    # shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    # split to batches
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def initialize_model(num_classes):
    print("===========================initialize model========================")
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

    # download & load the layer as a feature vector
    keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)

    model = tf.keras.Sequential([
        keras_layer,
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    # build the model with input image shape as (64, 64, 3)
    model.build([None, 64, 64, 3])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tfa.metrics.F1Score(num_classes = 10)]
    )
    return model

def train_model(model, num_examples):
    print("===========================train model========================")

    model_name = "satellite-classification"
    model_path = os.path.join("results", model_name + ".h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)

    # set the training & validation steps since we're using .repeat() on our dataset
    # number of training steps
    n_training_steps   = int(num_examples * 0.6) // batch_size
    # number of validation steps
    n_validation_steps = int(num_examples * 0.2) // batch_size
    # train the model
    history = model.fit(
        train_ds, validation_data=valid_ds,
        steps_per_epoch=n_training_steps,
        validation_steps=n_validation_steps,
        verbose=1, epochs=5,
        callbacks=[model_checkpoint]
)
    return history

if __name__=='__main__':
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset()

    batch_size = 64
    train_ds = prepare_for_training(train_ds, batch_size=batch_size)

    valid_ds = prepare_for_training(valid_ds, batch_size=batch_size)

    model = initialize_model(num_classes)

    history = train_model(model, num_examples)

    print("=======================end training model=======================")
