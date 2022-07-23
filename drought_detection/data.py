# python script that handles data (retrieved from the cloud)
import numpy as np
import tensorflow as tf
from google.cloud import storage

# example file from gs bucket:
# gs://wagon-data-batch913-drought_detection/data/train/part-r-00090

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
    imgArray = np.zeros((65,65,n_bands), 'uint8') # create empty array

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


def get_ndvi(imgArray):
    '''
    this function computes the ndvi image (2D array of 65x65 pixels)
    from an satellite image FILE

    Parameters:
            image_file (path): filename of image (including path)

    Returns:
            ndvi (list): a 2D (65x65 pixels) arrays of values
            label (int): label

    '''
    # Step 1-a: cast to float32 so we can do normal matrix computations (uint8 returns to 0 after 255)
    swapped_img = imgArray.insert(0, imgArray.pop()) # moves last element to front (does not swap first and last, which messes up order)
    cast_image = tf.cast(swapped_img, 'float32')

    # Step 1-b: calculate composite band (e.g. NDVI)
    ndvi_tmp = (cast_image[1] - cast_image[0]) / (cast_image[1] + cast_image[0])

    # Step 1-d: correct swapped index
    ndvi = ndvi_tmp[1:] + [ndvi_tmp[0]]

    return ndvi

# function to load single file (can be from 'gs:://BUCKET')
def load_single_img(image_file='../raw_data/train/part-r-00090', bands=['B4', 'B3', 'B2'], intensify=True):
    '''loads a single image from a file, outputs an imgArray and label'''
    parsed_sat = read_sat_file(image_file, bands)
    imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
    return imgArray, label


# function to load a set of files from a directory
def load_imgs(n_files, bands=['B4', 'B3', 'B2'], intensify=True, dataset='train', add_ndvi=False):
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

    # list_ds = tf.data.Dataset.list_files('gs://wagon-data-batch913-drought_detection/data/val/*')

    for filename in filenames_suffix:
        file = f'gs://wagon-data-batch913-drought_detection/{filename}'
        parsed_sat = read_sat_file(file, bands)
        imgArray, label = transform_sat_img(parsed_sat, bands, intensify)
        # ##### TO DO: ADD COMPOSITE BAND CALCULATIONS HERES #####
        # if add_ndvi=True:
        #     ndvi = get_ndvi(imgArray)
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
    # # convert labels to correct type
    # labels = np.array(labels)
    # labels = labels.astype('int64')

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
                 bands=['B4', 'B3', 'B2'], intensify=True):
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
    filenames, images, labels = load_imgs(train_n, bands, intensify, dataset='train_full')
    train_ds = make_prefetch_dataset(filenames, images, labels)

    print("===================================loading validation=====================================")
    # validation data set (data to help create metrics)
    filenames_v, images_v, labels_v = load_imgs(val_n, bands, intensify, dataset='val')
    valid_ds = make_prefetch_dataset(filenames_v, images_v, labels_v)

    print("=====================================loading test========================================")
    # test data (data you DO NOT TOUCH! :P)
    filenames_t, images_t, labels_t = load_imgs(test_n, bands, intensify, dataset='test')
    test_ds = make_prefetch_dataset(filenames_t, images_t, labels_t)

    # total number of classes (4 in our case)
    num_classes = len(set(labels))
    # total number of images
    num_examples = len(filenames) + len(filenames_v) + len(filenames_t)

    return train_ds, test_ds, valid_ds, num_examples, num_classes



if __name__ == '__main__':
    # Load data
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset()

    print(num_classes)
