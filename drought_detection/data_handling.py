# python script that handles data (retrieved from the cloud)

import os
import numpy as np
import tensorflow as tf
from google.cloud import storage
import tempfile
# from drought_detection.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH


# list files in directory
def dirlist(directory):
    '''list files in directory'''
    return [os.path.join(directory, file) for file in os.listdir(directory) if 'part-' in file]

# function to read raw satellite file data into a list of tensor dictionaries
def read_sat_file(image_file, bands_):
    '''
    This function filters satellite image data by specific spectral bands
    The function loads a batch of satellite images from a list of files
    and parses the satellite image data files for some specific spectral bands
    (e.g. B2, B3, B4, see official documentation)

    Parameters:
            list of satellite image files (including path, e.g '/data/train/part-r-00000')
    Returns:
            list of dictionaries (tensors) of raw satellite data (filtered by spectral band)
    '''
    # make tfrecord format list for chosen bands
    tfrecord_format = {}
    for b in bands_:
        tfrecord_format[b] = tf.compat.v1.FixedLenFeature([], tf.string)
    tfrecord_format['label'] = tf.compat.v1.FixedLenFeature([], tf.int64)

    # load and parse one sat image
    dataset = tf.data.TFRecordDataset(image_file)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    parsed_sat_file = [tf.compat.v1.parse_single_example(data, tfrecord_format) for data in iterator]

    return parsed_sat_file


# function that converts a raw sat image (tensorflow object) to matrix of numbers & label (it also scales bands)
def transform_sat_img(parsed_sat_file, bands_=['B4', 'B3', 'B2'], intensify_=True):
    '''
    This function creates a 3D imgArray in shape 65 x 65 x n_bands (65x65 pixels) for
    a single parsed satellite image, while also scaling each spectral band.

    Parameters:
            parsed_sat_img (dict): a parsed satellite image: Specific Tensorflow format (as dictionary)
            bands (list): list of bands to process (order is important!)
            intensify (bool): whether to scale or not (affects how bright plotted image looks(?))

    Returns:
            imgArray (tuple): tuple of processed images in n-Dimensional arrays (depends on number of bands chosen)
            label (list): list of corresponding labels (as int32)
    '''
    # convert to image array of numbers and label
    n_bands = len(bands_) # number of of bands determines depth of imgArray
    imgArray = np.zeros((65,65,n_bands), 'uint8') # create empty array

    # transform, reshape, and intensity-scale image data
    for i, band in enumerate(bands_): # order of specified bands is important because that is the order they will be appended
        band_data = np.frombuffer(parsed_sat_file[0][band].numpy(), dtype=np.uint8) # transforms raw tensorflow data into 1D array
        band_data = band_data.reshape(65, 65) # reshapes data into 65 x 65 pixel matrix
        if intensify_:
            band_data = band_data/np.max(band_data)*255 # scaling digital numbers so image is slightly brighter
        else:
            band_data = band_data*255 # scaling digital numbers
        imgArray[..., i] = band_data

    label = tf.cast(parsed_sat_file[0]['label'], tf.int32).numpy() # gets label for image

    return imgArray, label


# function to load single file (can be from 'gs:://BUCKET')
def load_img(file='../raw_data/train/part-r-00001', bands=['B4', 'B3', 'B2'], intensify=True):
    parsed_sat1 = read_sat_file(file, bands_=bands)
    imgArray, label = transform_sat_img(parsed_sat1, bands_=bands, intensify_=intensify)
    return imgArray, label



# function to load a set of files from a directory
def load_imgs_set(directory='../raw_data/train/', n_files = 2, bands=['B4', 'B3', 'B2'], intensify=True):
    '''
    This function creates a list of 3D imgArrays and a list of corresponding labels for
    a set of satellite image files in a specific folder

    Parameters:
            directory (path): path to directory containing raw satellite data files (can be from gs:://BUCKET)
            n_files (int): number of files to parse and transform
            bands (list): list of bands to process (order is important!)
            intensify (bool): whether to scale or not (affects how bright plotted image looks(?))

    Returns:
            images (list): list of processed images (65x65 pixels each) in n-Dimensions (depends on number of bands chosen)
            labels (list): list of corresponding labels (as int32)
    '''
    images = []
    labels = []

    files = dirlist(directory)

    for n in range(n_files):
        imgArray, label = load_img(file=files[n], bands=bands, intensify=intensify)
        images.append(imgArray)
        labels.append(label)

    return images, labels



# function to download images
def get_images_gcp(n=2, data_set='train', bands=['B4', 'B3', 'B2']):
    '''
    This function gets images from the cloud in the correct format.
    The function downloads images into temporary files, does a transformation, and then deletes the temporary file.

    Parameters:
            n (int): number of satellite images to process
            data_set (str): data folder to select images ('train' or 'val')
            bands (list): list of bands to process (order is important)

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
        img_sat, img_label = load_img(file=temp_local_filename, bands=bands, intensify=True)
        # append image (as 3D matrix) and label to lists
        images.append(img_sat)
        labels.append(img_label)
        # remove temporary file
        os.remove(temp_local_filename)

    return images, labels



if __name__ == '__main__':
    # train_imgs, train_labels = load_imgs_set(directory="gs://wagon-data-batch913-drought_detection/data/train/",
    #                              n_files=5,
    #                              bands=['B4', 'B3', 'B2'])

    # val_imgs, val_labels = load_imgs_set(directory="gs://wagon-data-batch913-drought_detection/data/val/",
    #                              n_files=5,
    #                              bands=['B4', 'B3', 'B2'])

    train_imgs, train_labels = get_images_gcp()
    val_imgs, val_labels = get_images_gcp(data_set='val')

    print(val_labels)
