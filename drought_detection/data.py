#!/usr/bin/env python3

# keras_train.py
# --------------------
# Use Keras to train a simple CNN to predict a discrete
# indicator of forage quality (inversely related to drought severity) from satellite
# images in 10 frequency bands. The ground truth label is the number of
# cows that a human expert standing at the center of the satellite image at ground level
# thinks the surrounding land could support (0, 1, 2, or 3+)


import argparse
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
INPUT_BANDS = ['all']   # input bands to use for training (composite, rgb, all, or custom list)
NUM_BANDS = 11          # number of bands being used (all = 11, composite = 3, rgb = 3)
DATA_PATH = "data"
BATCH_SIZE = 64


# fixed variables (do not alter!)
# for categorical classification, there are 4 classes: 0, 1, 2, or 3+ cows
NUM_CLASSES = 4
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


# raw file-loading
#----------------------------------
def load_files(data_path='data', local=True):
    # gcp bucket: 'wagon-data-batch913-drought_detection/data'
    if not local:
        data_path = f'gs://{data_path}'

    train_tfrecords = tf.data.Dataset.list_files(f"{data_path}/train/part*")
    val_tfrecords = tf.data.Dataset.list_files(f"{data_path}/val/part*")
    return train_tfrecords, val_tfrecords

# band data decoding
#----------------------------------
def decode_band(example, band):
    decoded_band = tf.io.decode_raw(example[band], tf.uint8)
    reshaped_band = tf.reshape(decoded_band[:IMG_DIM**2],
                            shape=(IMG_DIM, IMG_DIM, 1))
    return reshaped_band

def select_bands(example, list_of_bands = ['B4', 'B3', 'B2']):
    # decode and reshape multi-band image
    reshaped_bands = [decode_band(example, band) for band in list_of_bands]
    # combine bands into tensor
    example['image'] = tf.concat(reshaped_bands, -1)
    return example

def select_all_bands(example):
    # declare bands
    list_of_bands = ['B1', 'B2', 'B3', 'B4', 'B5', \
        'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
    # decode and reshape multi-band image
    reshaped_bands = [decode_band(example, band) for band in list_of_bands]
    # combine bands into tensor
    example['image'] = tf.concat(reshaped_bands, -1)
    return example

def select_rgb_bands(example):
    # declare bands
    list_of_bands = ['B4', 'B3', 'B2']
    # decode and reshape multi-band image
    reshaped_bands = [decode_band(example, band) for band in list_of_bands]
    # combine bands into tensor
    example['image'] = tf.concat(reshaped_bands, -1)
    return example

def compute_composites(example):
    # declare base bands used for computations
    bands = ['B2', 'B3', 'B4', 'B5']

    # convert values to float (to be able to to computations)
    values = [decode_band(example, band).astype('float32') for band in bands]
    floated_bands = dict(zip(bands, values))

    # compute ndvi
    ndvi = (floated_bands['B5'] - floated_bands['B4']) / \
        (floated_bands['B5'] + floated_bands['B4'])

    # compute arvi
    arvi = (floated_bands['B5'] - (2*floated_bands['B4']) + floated_bands['B2']) / \
    (floated_bands['B5'] + (2*floated_bands['B4']) + floated_bands['B5'])

    # compute gci
    gci = (floated_bands['B5']/floated_bands['B3']) - 1

    return [ndvi, arvi, gci]

def select_composite_bands(example):
    # compute composites
    composites = compute_composites(example)
    # combine composite bands into tensor
    example['image'] = tf.concat(composites, -1)
    return example

# image resizing/augmentation
#----------------------------------
image_resize = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(IMG_RESIZE, IMG_RESIZE)
        ],
        name="resize_layer"
    )

image_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(factor=0.1)
        ],
        name="augmentation_layer"
    )

# data parsing functions
#----------------------------------
def parse_tfrecord(unparsed_example):

    # declare feature descriptions/types
    feature_description = {
        'B1': tf.io.FixedLenFeature([], tf.string),
        'B2': tf.io.FixedLenFeature([], tf.string),
        'B3': tf.io.FixedLenFeature([], tf.string),
        'B4': tf.io.FixedLenFeature([], tf.string),
        'B5': tf.io.FixedLenFeature([], tf.string),
        'B6': tf.io.FixedLenFeature([], tf.string),
        'B7': tf.io.FixedLenFeature([], tf.string),
        'B8': tf.io.FixedLenFeature([], tf.string),
        'B9': tf.io.FixedLenFeature([], tf.string),
        'B10': tf.io.FixedLenFeature([], tf.string),
        'B11': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        }

    # parse example using feature descriptions
    example = tf.io.parse_single_example(unparsed_example, feature_description)

    return example

def encode_label(example):
    # one-hot encode label
    example['label'] = tf.cast(example['label'], tf.int32)
    example['label'] = tf.one_hot(example['label'], NUM_CLASSES)
    return example

def compute_empty(example):
    # sum value of all bands
    example['is_empty'] = tf.math.reduce_sum(example['image'])
    return example

def select_image_label(example):
    return example['image'], example['label']


# putting it all together
#----------------------------------
def get_dataset(tfrecords, input_bands):
    '''
    This function parses the TF Records, selects specific bands, encodes label,
    removes null images, and returns a TFDataset of images and labels

    Parameters:
            tfrecords (Dataset): Tensorflow ShuffleDataset of strings of file names
            args (args): object of parsed arguments from the command line
    Returns:
            dataset (Dataset): Tensorflow ParallelMapDataset of (images, labels)
    '''

    # parse TF records
    dataset = (
        tfrecords.interleave(tf.data.TFRecordDataset)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    )

    # select input bands
    if input_bands == ['all']:
        dataset = dataset.map(select_all_bands,
                              num_parallel_calls=AUTOTUNE)
        print("=========================loaded all bands========================")

    elif input_bands == ['composite']:
        dataset = dataset.map(lambda x: select_composite_bands(x),
                              num_parallel_calls=AUTOTUNE)
        print("=====================loaded composite bands======================")

    elif input_bands == ['rgb']:
        dataset = dataset.map(select_rgb_bands,
                              num_parallel_calls=AUTOTUNE)
        print("=========================loaded rgb bands========================")

    else:
        dataset = dataset.map(lambda x: select_bands(x, input_bands),
                              num_parallel_calls=AUTOTUNE)
        print(f"=================loaded {input_bands} bands==================")

    # encode labels, remove bad images, return only image and label
    dataset = (
        dataset
            .map(encode_label, num_parallel_calls=AUTOTUNE)     # one-hot encode labels
            .map(compute_empty, num_parallel_calls=AUTOTUNE)    # tag bad images
            .filter(lambda example: example['is_empty'] != 0)    # remove bad images
            .map(select_image_label, num_parallel_calls=AUTOTUNE) # return only (images, labels)
            .cache() # cache data to save previous operations from being executed during each epoch
        )

    return dataset

def prepare_dataset(ds, batch_size, shuffle_size=NUM_TRAIN, shuffle=False, augment=False):

    # Shuffle only the training set
    if shuffle:
        ds = ds.shuffle(shuffle_size)

    # Batch all datasets.
    ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)

    # Resize all datasets.
    ds = ds.map(lambda x, y: (image_resize(x), y),
                num_parallel_calls=AUTOTUNE)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (image_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use repeat and buffered prefetching on all datasets.
    return ds.repeat(-1).prefetch(buffer_size=AUTOTUNE)

def load_dataset(data_path='data',
                 input_bands=['all'],
                 batch_size=BATCH_SIZE,
                 train_shuffle_size=NUM_TRAIN):
    # load training data in TFRecord format
    train_tfrecords, val_tfrecords = load_files(data_path)

    # load correct multi-band images and labels from TFRecords
    train_images_labels = get_dataset(train_tfrecords, input_bands)
    val_images_labels = get_dataset(val_tfrecords, input_bands)

    # prepare dataset for training
    train_dataset = prepare_dataset(train_images_labels, batch_size,
                                    train_shuffle_size, shuffle=True)
    val_dataset = prepare_dataset(val_images_labels, batch_size)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = load_dataset(batch_size=1, train_shuffle_size=1)
    train_image, train_label = next(iter(train_dataset))
    print(train_image, train_label)
