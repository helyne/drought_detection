# python script that handles data (retrieved from the cloud)

import os
import numpy as np
import tensorflow as tf
from google.cloud import storage
import tempfile
# import cv2
from drought_detection.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH


# function that selects bands (all in this case) and gets raw data as tensorflow object
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


# function that converts a raw sat image (tensorflow object) to matrix of numbers & label (it also scales bands)
def get_img_from_example(parsed_sat_img, bands=['B4', 'B3', 'B2'], intensify=True):
    '''
    This function creates an RGB 3D array in shape 65x65x3 (65x65 pixels) for
    a single parsed satellite image, while also scaling each spectral band.

    For each band (depends on filtering done by above function),
    the raw band data is transformed from the Tensorflow specific data format into
    a 2D array of dimension 65x65 pixels. In this case, it orders the
    bands by red, green, blue (for plotting presumably)

    Next, it does some scaling: if intensity=True, it divides each pixel by the
    maximum value of the pixels, and then multiplies it by 255.
    Otherwise, if intensify=False, it just multiplies each pixel in the matrix by 255.
    "When the range of pixel brightness values is closer to 0, a darker image is rendered by default.
    You can stretch the values to extend to the full 0-255 range of potential values to
    increase the visual contrast of the image."
    - https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/

    Lastly, the function adds the corresponding label, and returns the image 2D array, as well as the label.

    Parameters:
            parsed_sat_img (dict) = a parsed satellite image: Specific Tensorflow format (as dictionary)
            bands (list): list of bands to process (order is important!)
            intensify (bool): whether to scale or not (affects how bright plotted image looks(?))

    Returns:
            rgbArray (tuple): tuple of processed images in n-Dimensional arrays (depends on number of bands chosen)
            label (list): list of corresponding labels (as int32)
    '''
    n_bands = len(bands) # number of of bands determines depth of rgbArray
    rgbArray = np.zeros((65,65,n_bands), 'uint8') # create empty array

    for i, band in enumerate(bands): # order of specified bands is important because that is the order they will be appended
        band_data = np.frombuffer(parsed_sat_img[band].numpy(), dtype=np.uint8) # transforms raw tensorflow data into 1D array
        band_data = band_data.reshape(65, 65) # reshapes data into 65 x 65 pixel matrix
        if intensify:
            band_data = band_data/np.max(band_data)*255 # scaling digital numbers so image is slightly brighter
        else:
            band_data = band_data*255 # scaling digital numbers
        rgbArray[..., i] = band_data

    label = tf.cast(parsed_sat_img['label'], tf.int32).numpy() # gets label for image

    return rgbArray, label



def get_images_gcp(n=1, data_set='train'):
    '''
    This function gets images from the cloud in the correct format.
    The function downloads images into temporary files, does a transformation, and then deletes the temporary file.

    Parameters:
            n (int): number of satellite images to process
            data_set (str): data folder to select images ('train' or 'val')

    Returns:
            images (list): list of processed images in RGB 3D arrays
            labels (list): list of corresponding labels (as int32)

    '''

    # GCP bucket parameters
    project_name = 'drought-detection'
    bucket_name = 'wagon-data-batch913-drought_detection'
    prefix = 'data/' + data_set + '/part'

    # open client and get blobs
    storage_client = storage.Client(project=project_name)
    blobs = storage_client.list_blobs(bucket_name,
                                      prefix=prefix, # what folder & filename prefix
                                      delimiter='/', # don't include subdirectories
                                      max_results=n) # max number of blobs to grab

    images = []
    labels = []

    for blob in blobs:
        # create temporary file
        _, temp_local_filename = tempfile.mkstemp()

        # Download blob from bucket into temp file
        blob.download_to_filename(temp_local_filename)

        # Do stuff to file (transform data format)
        parsed_img = parse_visual_rgb(temp_local_filename) # parse satellite image data
        img_sat, img_label = get_img_from_example_rgb(parsed_img[0]) # convert data to rgbArray with label

        # append image (as 3D matrix) and label to lists
        images.append(img_sat)
        labels.append(img_label)

        # remove temporary file
        os.remove(temp_local_filename)

    return np.array(images), labels





































def parse_visual_rgb(data):
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
        'B2': tf.compat.v1.FixedLenFeature([], tf.string), # blue spectral band
        'B3': tf.compat.v1.FixedLenFeature([], tf.string), # green spectral band
        'B4': tf.compat.v1.FixedLenFeature([], tf.string), # red spectral band
        'label': tf.compat.v1.FixedLenFeature([], tf.int64), # image label (0/1/2/3)
    }

    parsed_sat_imgs = [tf.compat.v1.parse_single_example(data, features) for data in iterator]
    return parsed_sat_imgs



def get_img_from_example_rgb(parsed_sat_img, intensify=True):
    '''
    This function creates an RGB 3D array in shape 65x65x3 (65x65 pixels) for
    a single parsed satellite image, while also scaling each spectral band.

    For each band (depends on filtering done by above function),
    the raw band data is transformed from the Tensorflow specific data format into
    a 2D array of dimension 65x65 pixels. In this case, it orders the
    bands by red, green, blue (for plotting presumably)

    Next, it does some scaling: if intensity=True, it divides each pixel by the
    maximum value of the pixels, and then multiplies it by 255.
    Otherwise, if intensify=False, it just multiplies each pixel in the matrix by 255.
    "When the range of pixel brightness values is closer to 0, a darker image is rendered by default.
    You can stretch the values to extend to the full 0-255 range of potential values to
    increase the visual contrast of the image."
    - https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/

    Lastly, the function adds the corresponding label, and returns the image 2D array, as well as the label.


    Input: - a parsed satellite image: Specific Tensorflow format (as dictionary)
    Output(s) - satellite image & its label:
                - rgbArray: tuple of 3D numpy array (shape 65x65x3)
                - label: int32
    '''
    rgbArray = np.zeros((65,65,3), 'uint8')
    for i, band in enumerate(['B4', 'B3', 'B2']): # order is important here
        band_data = np.frombuffer(parsed_sat_img[band].numpy(), dtype=np.uint8) # transforms raw tensorflow data into 1D array
        band_data = band_data.reshape(65, 65) # reshapes data into 65 x 65 pixel matrix
        if intensify:
            band_data = band_data/np.max(band_data)*255 # are we transforming the image from bytes to digital numbers by multiplying by 255?
        else:
            band_data = band_data*255
        rgbArray[..., i] = band_data

    label = tf.cast(parsed_sat_img['label'], tf.int32).numpy() # gets label for image

    return rgbArray, label






if __name__ == '__main__':
    image, label = get_images_gcp()
