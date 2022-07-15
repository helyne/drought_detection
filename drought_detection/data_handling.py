# python script that handles data (retrieved from the cloud)
import numpy as np
import tensorflow as tf
from google.cloud import storage


# list files in directory from google storage
def dirlist(dataset='train'):
    '''list files in directory'''
    client = storage.Client()
    bucket = client.bucket('wagon-data-batch913-drought_detection')
    blobs = list(bucket.list_blobs(prefix=f'data/{dataset}/part'))
    return [blob_.name for blob_ in blobs]


# function to read raw satellite file data
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


# function to convert a raw sat image (tensorflow object) to matrix of numbers & label (it also scales bands)
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
def load_single_img(image_file='../raw_data/train/part-r-00001', bands=['B4', 'B3', 'B2'], intensify=True):
    '''loads a single image from a file, outputs an imgArray and label'''
    parsed_sat1 = read_sat_file(image_file, bands_=bands)
    imgArray, label = transform_sat_img(parsed_sat1, bands_=bands, intensify_=intensify)
    return imgArray, label


# function to load a set of files from a directory
def load_imgs(dataset='train', n_files = 2, bands=['B4', 'B3', 'B2'], intensify=True):
    '''
    This function creates a list of 3D imgArrays and a list of corresponding labels for
    a set of satellite image files in a specific folder

    Parameters:
            dataset (string): one of: 'train' | 'val' | 'test'
            n_files (int): number of files to parse and transform, max 319(train), 81(val), or 100(test)
            bands (list): list of bands to process (order is important!)
            intensify (bool): whether to scale or not (affects how bright plotted image looks(?))

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
        imgArray, label = load_single_img(file=file, bands=bands, intensify=intensify)
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
    AUTO = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices({'filename': filenames,
                                                'image': images,
                                                'label': labels})
    # dataset = dataset.shuffle(2048) # we shuffle later, not sure we need it here
    dataset = dataset.prefetch(AUTO)

    return dataset


# final function to load data (read, convert, transform)
def load_dataset(train_n = 1, val_n = 1, test_n = 1,
                 bands=['B4', 'B3', 'B2']):
    '''
    This function loads train, validation, and test datasets from GCP

    Parameters:
            train_n/val_n/test_n (int): number of image files to load for each dataset
            bands (list): list of bands to process (order is important!)

    Returns:
            train_ds (PrefetchDataset): train data
            valid_ds (PrefetchDataset): validation data
            test_ds (PrefetchDataset): test data
            num_examples (int): total number of images (train + val + test)
            num_classes (int):

    '''
    print("====================================loading dataset======================================")

    # load training, testing & validation sets
    # train data set
    filenames, images, labels = load_imgs_set(dataset='train',
                                              n_files = train_n,
                                              bands=bands,
                                              intensify=True)
    train_ds = make_prefetch_dataset(filenames, images, labels)
    # validation data set (data to help create metrics)
    filenames_v, images_v, labels_v = load_imgs_set(dataset='val',
                                              n_files = val_n,
                                              bands=bands,
                                              intensify=True)
    valid_ds = make_prefetch_dataset(filenames_v, images_v, labels_v)
    # test data (data you DO NOT TOUCH! :P)
    filenames_t, images_t, labels_t = load_imgs_set(dataset='test',
                                              n_files = test_n,
                                              bands=bands,
                                              intensify=True)
    test_ds = make_prefetch_dataset(filenames_t, images_t, labels_t)

    # # the class names
    # class_names = [0, 1, 2, 3] # class values (eg. 0 cows, 1 cows, etc.)
    # total number of classes (4)
    num_classes = 4
    # total number of images we have
    num_examples = len(filenames) + len(filenames_v) + len(filenames_t)

    return train_ds, test_ds, valid_ds, num_examples, num_classes



if __name__ == '__main__':
    # Load data
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset()

    print(num_classes)
