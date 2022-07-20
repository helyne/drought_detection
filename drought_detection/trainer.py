import os
from pydoc import doc
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa
# import datetime

from drought_detection.data_handling import load_dataset
from google.cloud import storage
import joblib



def prepare_for_training(ds, num_classes, cache=True, batch_size=64, shuffle_buffer_size=1000):
    print("===========================starting preprocessing========================")
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    # one hot encode classes
    ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
    # shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    # split to batches
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training. Autotune automatically sets the appropriate buffer size
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def initialize_model(num_classes):
    print("===========================initialize model========================")
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

    # download & load the layer as a feature vector
    keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=False)

    model = tf.keras.Sequential([
        keras_layer,
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    # build the model with input image shape as (65, 65, 3)
    # shape = train_ds.take(1)['image'].shape
    # shape = [None, shape]
    model.build([None, 65, 65, 3]) # (placeholder for num images, 65 pixel width, 65 pixel height, 3 bands)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tfa.metrics.F1Score(num_classes = num_classes)]
    )
    return model



def train_model(model, num_examples):
    print("===========================train model========================")

    # setup save, checkpoint, and weights paths
    model_path = "gs://wagon-data-batch913-drought_detection/classification_helyne_test"
    weights_file = "gs://wagon-data-batch913-drought_detection/weights.h5"
    checkpoint_path = "gs://wagon-data-batch913-drought_detection/classification_helyne_testsave_at_{epoch}"

    # set model callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
        # tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1), # if we want pretty dashboard
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3), # if we want to use early stopping
    ]

    # set the training & validation steps since we're using .repeat() on our dataset
    # number of training steps
    n_training_steps   = int(num_examples * 0.6) / batch_size #it was 0.6 before
    # number of validation steps
    n_validation_steps = int(num_examples * 0.2) / batch_size #it was 0.2 before

    # train the model
    history = model.fit(
        train_ds, validation_data=valid_ds,
        steps_per_epoch=n_training_steps,
        validation_steps=n_validation_steps,
        verbose=1, epochs=50, #it was 5 before
        callbacks=callbacks
    )

    # set storage location in cloud
    STORAGE_LOCATION = 'SavedModel/RGB_50epo_8batch_fullset_trainFalse_ps'

    # save directly to gcp
    model.save(f'gs://wagon-data-batch913-drought_detection/{STORAGE_LOCATION}') #not working
    #tf.keras.models.save_model(model, f'gs://wagon-data-batch913-drought_detection/{STORAGE_LOCATION}', save_format="tf")
    print(f"uploaded model to gcp cloud storage under \n => {STORAGE_LOCATION}")
    print("===========================saved model========================")
    return history, model_path, weights_file

def upload_model_to_gcp(modelfile_name, STORAGE_LOCATION):

    BUCKET_NAME='wagon-data-batch913-drought_detection'

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename(modelfile_name)


def evaluate_model(model_path, num_examples):
    print("=======================Starting evaluation=======================")
    # load the model
    model = joblib.load(model_path)

    # number of testing steps
    n_testing_steps = int(num_examples * 0.6) #it was 0.6 before
    # get all testing images as NumPy array
    images = np.array([ d["image"] for d in test_ds.take(n_testing_steps) ])
    print("images.shape:", images.shape)
    # get all testing labels as NumPy array
    labels = np.array([ d["label"] for d in test_ds.take(n_testing_steps) ])
    print("labels.shape:", labels.shape)
    # feed the images to get predictions
    predictions = model.predict(images)
    # perform argmax to get class index
    predictions = np.argmax(predictions, axis=1)
    print("predictions.shape:", predictions.shape)
    # evaluate model
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(labels, predictions)
    print("Accuracy:", accuracy.result().numpy())
    print("F1 Score:", f1_score(labels, predictions, average="macro"))
    return accuracy



if __name__ == '__main__':
    # Load data
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset(bands=['B4', 'B3', 'B2'])

    print("=======================Load dataset=======================")

    batch_size = 8 #should be 64

    train_ds = prepare_for_training(train_ds, num_classes, batch_size=batch_size)

    valid_ds = prepare_for_training(valid_ds, num_classes, batch_size=batch_size)

    print("=======================Initialize model=======================")

    model = initialize_model(num_classes)

    print("=======================Train model=======================")

    history, model_path, weights_file = train_model(model, num_examples)

    # model_path = 'model.joblib'
    # accuracy  = evaluate_model(model_path, num_examples)

    # print("=======================Evaluate Model=======================")
