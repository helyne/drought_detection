import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import save_model

def load_dataset():
    print("======================================starting======================================")
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path="mnist.npz")

    X_train = X_train / 255.
    X_test = X_test / 255.


    X_train = expand_dims(X_train, axis=-1)
    X_test = expand_dims(X_test, axis=-1)


    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    return X_train, y_train_cat

def initialize_model():
    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4,4), input_shape=(28, 28, 1), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with 10 outputs corresponding to 10 digits
    model.add(layers.Dense(10, activation='softmax'))

    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

#model = initialize_model()



# $CHALLENGIFY_BEGIN
def train_model(X_train, y_train_cat):
    print("======================================data loaded======================================")
    model = initialize_model()

    es = EarlyStopping(patience = 5)

    history = model.fit(X_train,
                y_train_cat,
                validation_split = 0.3,
                batch_size = 32,
                epochs = 5,
                callbacks = [es],
                verbose = 1)
    print("======================================model trained======================================")

    return history

def save_model(model):
    """Save the model"""
    save_model(model, "model.h5")
    print("Saved model to disk")


'''

def save_model(self):
    """Save the model"""
    model.save("model.h5")
    print("Saved model to disk")


def save_model_to_gcp(reg):
    """Save the model into a .joblib and upload it on Google Storage /models folder
    HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
    from sklearn.externals import joblib
    local_model_name = 'model.joblib'
    # saving the trained model to disk (which does not really make sense
    # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
    joblib.dump(reg, local_model_name)
    print("saved model.joblib locally")
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))

    https://machinelearningmastery.com/save-load-keras-deep-learning-models/

    =====> We need to add save_model from tensorflow.keras.models, and we have to incorporated to the function above

'''

if __name__=='__main__':
    print("======================================starting======================================")
    X_train, y_train_cat = load_dataset()
    print("======================================data loaded======================================")
    print(train_model(X_train, y_train_cat))
    print("======================================model trained======================================")
