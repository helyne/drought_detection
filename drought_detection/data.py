# python script that handles data (retrieved from the cloud)
import numpy as np
import tensorflow as tf
from google.cloud import storage

# example file from gs bucket:
# gs://wagon-data-batch913-drought_detection/data/train/part-r-00090

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

# bands_list = [ 'B1', 'B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']


# list files in directory from google storage
def dirlist(dataset='train'):
    '''list files in directory'''
    client = storage.Client()
    bucket = client.bucket('wagon-data-batch913-drought_detection')
    blobs = list(bucket.list_blobs(prefix=f'data/{dataset}/part'))
    return [blob_.name for blob_ in blobs]


# function to read raw satellite file data
def read_sat_file(image_file, bands):
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
    for b in bands:
        tfrecord_format[b] = tf.io.FixedLenFeature([], tf.string)
    tfrecord_format['label'] = tf.io.FixedLenFeature([], tf.int64)

    # load and parse one sat image
    dataset = tf.data.TFRecordDataset(image_file)
    iterator = iter(dataset)
    parsed_sat_file = [tf.io.parse_single_example(data, tfrecord_format) for data in iterator]

    return parsed_sat_file


# function to convert a raw sat image (tensorflow object) to matrix of numbers & label (it also scales bands)
def transform_sat_img(parsed_sat_file, bands, intensify=True):
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
    n_bands = len(bands) # number of of bands determines depth of imgArray
    imgArray = np.zeros((65,65,n_bands), 'uint64') # create empty array

    # transform, reshape, and intensity-scale image data
    for i, band in enumerate(bands): # order of specified bands is important because that is the order they will be appended
        band_data = tf.io.decode_raw(parsed_sat_file[0][band], tf.uint8) # transforms raw tensorflow data into 1D array
        band_data = tf.reshape(band_data, [65,65]) # reshapes data into 65 x 65 pixel matrix
        if intensify:
            band_data = band_data/np.max(band_data)*255 # scaling digital numbers so image is slightly brighter
        else:
            band_data = band_data*255 # scaling digital numbers
        imgArray[..., i] = band_data

    label = tf.cast(parsed_sat_file[0]['label'], tf.int32).numpy() # gets label for image

    return imgArray, label


############################ Composite Band Functions ####################################

#Most up to date code for ARVI #LOAD B2, B4 AND B5 in this order!!
def get_arvi(imgArray):
    # swap axes (we needed to index into the list of lists)
    swapped = imgArray.swapaxes(0,-1)
    # make float (to do calculations)
    cast_image = swapped.astype('float32')
    # calculate composite band (e.g. here, GCI)
    arvi = (cast_image[2] - (2 * cast_image[1]) + cast_image[0]) / (cast_image[2] + (2 *cast_image[1]) + cast_image[0])
    return arvi


#Most up to date code for NDVI #LOAD B4 AND B5 in this order!!
def get_ndvi(imgArray):
    # swap axes (we needed to index into the list of lists)
    swapped = imgArray.swapaxes(0,-1)
    # make float (to do calculations)
    cast_image = swapped.astype('float32')
    # calculate composite band (e.g. here, GCI)
    ndvi = (cast_image[1] - cast_image[0]) / (cast_image[1] + cast_image[0])
    return ndvi

#Most up to date code for GCI #LOAD B3 AND B5 in this order!!
def get_gci(imgArray):
    # swap axes (we needed to index into the list of lists)
    swapped = imgArray.swapaxes(0,-1)
    # make float (to do calculations)
    cast_image = swapped.astype('float32')
    # calculate composite band (e.g. here, GCI)
    gci = (cast_image[1]/cast_image[0])-1
    return gci

# Stack 1-dimension GCI array and 2 empty dimensions to match the model input format requirements
def dummy_dim(input=None, dummy_1=np.zeros((65,65)), dummy_2=np.zeros((65,65))):
    return np.dstack([input, dummy_1, dummy_2])

##########################################################################################


# function to load single file (can be from 'gs:://BUCKET')
def load_single_img(image_file='../raw_data/train/part-r-00090', bands=['B4', 'B3', 'B2'], intensify=True):
    '''loads a single image from a file, outputs an imgArray and label'''
    parsed_sat = read_sat_file(image_file, bands)
    imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
    return imgArray, label


# function to load a set of files from a directory
def load_imgs(n_files, bands=['B4', 'B3', 'B2'], intensify=True, dataset='train', composite=''):
    '''
    This function creates a list of 3D imgArrays and a list of corresponding labels for
    a set of satellite image files in a specific folder

    Parameters:
            n_files (int): number of files to parse and transform, max 319(train), 81(val), or 100(test)
            bands (list): list of bands to process (order is important!)
            intensify (bool): whether to scale or not (affects how bright plotted image looks(?))
            dataset (string): one of: 'train' | 'val' | 'test'
            composite (string): one of 'gci' | 'ndvi' | 'arvi'
    Returns:
            images (list): list of processed images (65x65 pixels each) in n-Dimensions (depends on number of bands chosen)
            labels (list): list of corresponding labels (as int32)
    '''

    filenames = []
    images = []
    labels = []

    filenames_suffix = dirlist(dataset)[0:n_files]

    for filename in filenames_suffix:
        file = f'gs://wagon-data-batch913-drought_detection/{filename}'
        if composite == 'gci':
            bands=['B3', 'B5']
            parsed_sat = read_sat_file(file, bands)
            imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
            gci_tmp = get_gci(imgArray)
            imgArray = dummy_dim(gci_tmp)
        elif composite == 'ndvi':
            bands=['B4', 'B5']
            parsed_sat = read_sat_file(file, bands)
            imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
            tmp = get_ndvi(imgArray)
            imgArray = dummy_dim(tmp)
        elif composite == 'arvi':
            bands = ['B2', 'B4', 'B5']
            parsed_sat = read_sat_file(file, bands)
            imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
            tmp = get_arvi(imgArray)
            imgArray = dummy_dim(tmp)
        else:
            parsed_sat = read_sat_file(file, bands)
            imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
        filenames.append(file)
        images.append(imgArray)
        labels.append(label)
    return filenames, images, labels


# function to transform images into PrefetchDataset
def make_prefetch_dataset(filenames, images, labels):
    '''
    This function transforms our data into the correct data structure for modelling.

    Parameters: (takes output directly from load_imgs_set())
            filenames (list): a list of satellite files
            images (tuple): tuple of processed images in n-Dimensional arrays (depends on number of bands chosen)
            label (list): list of corresponding labels (as int32)

    Returns:
            dataset (PrefetchDataset):
                <PrefetchDataset shapes: {filename: (), image: (65, 65, 3), label: ()},
                types: {filename: tf.string, image: tf.uint8, label: tf.int64}>
    '''

    # create Dataset from data
    dataset = tf.data.Dataset.from_tensor_slices({'filename': filenames,
                                                'image': images,
                                                'label': labels})

    # convert Dataset to PrefetchDataset
    # "If the value tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned"
    # AUTO = tf.data.experimental.AUTOTUNE
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# final function to load data (read, convert, transform)
def load_dataset(train_n=319, val_n=81, test_n=100,
                 bands=['B4', 'B3', 'B2'], intensify=True, composite=''):
    '''
    This function loads train, validation, and test datasets from GCP

    Parameters:
            train_n/val_n/test_n (int): number of image files to load for each dataset (max values as defaults)
            bands (list): list of bands to process (order is important!)

    Returns:
            train_ds (PrefetchDataset): train data
            valid_ds (PrefetchDataset): validation data
            test_ds (PrefetchDataset): test data
            num_examples (int): total number of images (train + val + test)
            num_classes (int):

    '''
    print("=====================================loading train========================================")
    # train data
    filenames, images, labels = load_imgs(train_n, bands, intensify, dataset='train_full', composite=composite)
    train_ds = make_prefetch_dataset(filenames, images, labels)

    print("===================================loading validation=====================================")
    # validation data set (data to help create metrics)
    filenames_v, images_v, labels_v = load_imgs(val_n, bands, intensify, dataset='val', composite=composite)
    valid_ds = make_prefetch_dataset(filenames_v, images_v, labels_v)

    print("=====================================loading test========================================")
    # test data (data you DO NOT TOUCH! :P)
    filenames_t, images_t, labels_t = load_imgs(test_n, bands, intensify, dataset='test', composite=composite)
    test_ds = make_prefetch_dataset(filenames_t, images_t, labels_t)

    # total number of classes (4 in our case)
    num_classes = len(set(labels))
    # total number of images
    num_examples = len(filenames) + len(filenames_v) + len(filenames_t)

    return train_ds, test_ds, valid_ds, num_examples, num_classes


if __name__ == '__main__':
    print("=======================Load dataset=======================")
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset(train_n=1, val_n=1, test_n=1,
                                                                          bands=['B2', 'B3', 'B4'])
    print(train_ds.take(1))

    # batch_size = 64 #should be 64
    # train_ds = prepare_for_training(train_ds, num_classes, batch_size=batch_size)
    # valid_ds = prepare_for_training(valid_ds, num_classes, batch_size=batch_size)
    # data_shape = list(train_ds.take(1).element_spec[0].shape)
    # print(data_shape)
