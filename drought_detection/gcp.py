# library imports
import os
from google.cloud import storage
from termcolor import colored
# imports from our parameters file
from drought_detection.params import BUCKET_NAME, MODEL_NAME


def storage_upload(bucket=BUCKET_NAME, rm=False):
    # open client to our bucket
    client = storage.Client().bucket(bucket) # may need to add parameter to storage.Client(project='drought-detection')

    # set storage location and file name for saved model
    storage_location = '{}/{}/{}'.format(
        'models',
        MODEL_NAME,
        'model.h5') # check this

    # save model to client
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.h5') # check this

    # print what just happened
    print(colored("=> model.h5 uploaded to bucket {} inside {}".format(
        BUCKET_NAME, storage_location), "blue"))

    # if rm=True, remove file
    if rm:
        os.remove('model.h5') # check this file name
