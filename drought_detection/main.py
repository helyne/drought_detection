from PIL import Image
import numpy as np
import tensorflow as tf

def transform_user_img(fileUpload):
    # open user image file
    image_uploaded = Image.open(fileUpload)

    # resize image
    image = image_uploaded.resize((65,65)) # we must resize, but will this be problematic? this limits the user input?

    # convert to np.array
    image = np.array(image)

    # cast to int64
    image = tf.cast(image, tf.int64) # do we need to make array??

    return image
