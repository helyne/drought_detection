#!/usr/bin/env python3

# keras_train.py
# --------------------
# Use Keras to train a simple CNN to predict a discrete
# indicator of forage quality (inversely related to drought severity) from satellite
# images in 10 frequency bands. The ground truth label is the number of
# cows that a human expert standing at the center of the satellite image at ground level
# thinks the surrounding land could support (0, 1, 2, or 3+)

import argparse
import math
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers, initializers
import plotly.express as px
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # solution to tensorflow version issues??? PLEASE WORK

# for categorical classification, there are 4 classes: 0, 1, 2, or 3+ cows
NUM_CLASSES = 4
# fixed example counts from full dataset in TFRecord format
TOTAL_TRAIN = 86317
TOTAL_VAL = 10778
# limited example counts for faster training/debugging
NUM_TRAIN = 16000
NUM_VAL = 3200

# default image side dimension (65 x 65 square)
IMG_DIM = 65
# use 7 out of 10 bands for now
NUM_BANDS = 7
# number of images to log (keep below 50 for best results)
NUM_LOG_IMAGES = 16


tf.compat.v1.set_random_seed(23)

# data-loading and parsing utils
#----------------------------------
def load_data(data_path):
  train = file_list_from_folder("train", data_path)
  val = file_list_from_folder("val", data_path)
  return train, val

def file_list_from_folder(folder, data_path):
  folderpath = os.path.join(data_path, folder)
  filelist = []
  for filename in os.listdir(folderpath):
    if filename.startswith('part-') and not filename.endswith('gstmp'):
      filelist.append(os.path.join(folderpath, filename))
  return filelist

# module-loading utils
#--------------------------------
def load_class_from_module(module_name):
  components = module_name.split('.')
  mod = __import__(components[0])
  for comp in components[1:]:
    mod = getattr(mod, comp)
  return mod

# def load_optimizer(optimizer, learning_rate):
#   """ Dynamically load relevant optimizer """
#   optimizer_path = "tensorflow.keras.optimizers." + optimizer
#   optimizer_module = load_class_from_module(optimizer_path)
#   return optimizer_module(lr=learning_rate)

# data field specification for TFRecords
features = {
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
  'label': tf.io.FixedLenFeature([], tf.int64),
}

def getband(example_key):
  img = tf.decode_raw(example_key, tf.uint8)
  return tf.reshape(img[:IMG_DIM**2], shape=(IMG_DIM, IMG_DIM, 1))

# returns a raw RGB image from the satellite image
def get_img_from_example(parsed_example, intensify=True):
  rgbArray = tf.zeros((65,65,3), 'uint8')
  bandlist = []
  for i, band in enumerate(['B4', 'B3', 'B2']):
    band_data = getband(parsed_example[band])
    band_data = tf.reshape(band_data, shape=(65, 65, 1))
    if intensify:
      band_data = band_data / tf.math.reduce_max(band_data)*255
    else:
      band_data = band_data*255
    bandlist.append(band_data)
  rgbArray = tf.concat(bandlist, -1)
  rgbArray = tf.reshape(rgbArray, shape=(IMG_DIM, IMG_DIM, 3))
  return rgbArray



def parse_visual(data):
    '''
    This function filters satellite image data by specific spectral bands (RGB in this case).
    The function loads a batch of satellite images from a list of files
    and parses the satellite image data files for some specific features,
    e.g. spectral bands (B2, B3, B4, see official documentation)

    Input(s): - list of satellite image files (including path, e.g '/data/train/part-r-00000')
    Outputs:  - list of dictionaries of raw satellite data (filtered by spectral band)
    '''
    dataset = tf.data.TFRecordDataset(data)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    features = {
        'B1': tf.compat.v1.FixedLenFeature([], tf.string),    # 0.43 - 0.45 μm Coastal aerosol
        'B2': tf.compat.v1.FixedLenFeature([], tf.string),    # Blue
        'B3': tf.compat.v1.FixedLenFeature([], tf.string),    # Green
        'B4': tf.compat.v1.FixedLenFeature([], tf.string),    # Red
        'B5': tf.compat.v1.FixedLenFeature([], tf.string),    # Near infrared
        'B6': tf.compat.v1.FixedLenFeature([], tf.string),    # Shortwave infrared 1
        'B7': tf.compat.v1.FixedLenFeature([], tf.string),    # Shortwave infrared 2
        'B8': tf.compat.v1.FixedLenFeature([], tf.string),
        'B9': tf.compat.v1.FixedLenFeature([], tf.string),
        'B10': tf.compat.v1.FixedLenFeature([], tf.string),
        'B11': tf.compat.v1.FixedLenFeature([], tf.string),
        'label': tf.compat.v1.FixedLenFeature([], tf.int64),
        }

    parsed_sat_imgs = [tf.compat.v1.parse_single_example(data, features) for data in iterator]
    return parsed_sat_imgs



def parse_tfrecords(filelist, batch_size, buffer_size, include_viz=False):
  # try a subset of possible bands
  def _parse_(serialized_example, keylist=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']):
    example = tf.io.parse_single_example(serialized_example, features)

    def getband(example_key):
      img = tf.compat.v1.decode_raw(example_key, tf.uint8)
      return tf.reshape(img[:IMG_DIM**2], shape=(IMG_DIM, IMG_DIM, 1))

    bandlist = [getband(example[key]) for key in keylist]

    # combine bands into tensor
    image = tf.concat(bandlist, -1)

    # # alternative method:
    # import xarray as xr
    # landsat_post_fire_xr = xr.concat(all_bands, dim="band")
    # landsat_post_fire_xr

    plt.imshow(image)
    plt.show()

    # one-hot encode ground truth labels
    label = tf.cast(example['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    # if logging RGB images as examples, generate RGB image from 11-channel satellite image
    if include_viz:
      image = get_img_from_example(example)
      return {'image' : image, 'label': example['label']}, label
    return {'image': image}, label

  tfrecord_dataset = tf.data.TFRecordDataset(filelist)
  # tensor.numpy
  tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x)).shuffle(buffer_size).repeat(-1).batch(batch_size)
  tfrecord_iterator = tf.compat.v1.data.make_one_shot_iterator(tfrecord_dataset)
  image, label = tfrecord_iterator.get_next()
  return image, label



# import plotly.express as px
# import numpy as np
# img_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#                     [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
#                    ], dtype=np.uint8)
# fig = px.imshow(img_rgb)
# fig.show()



if __name__ == '__main__':
    # create a function called dirlist that extracts a list of file names (file) from a directory (di)
    dirlist = lambda di: [os.path.join(di, file) for file in os.listdir(di) if 'part-' in file]
    print(dirlist)

    # get list of files
    validation_files = dirlist('./raw_data/val/')
    print(validation_files)

    images = parse_tfrecords(validation_files, 2, 2, include_viz=True)

    # print(images.shape)

    a=1
