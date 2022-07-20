import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from drought_detection.gcp import storage_upload
import io
import tensorflow_hub as hub
import joblib

# # Load the model (only executed once!)
# @st.cache
# def load_model():
#     return joblib.load('SavedModel_model.joblib')
#     # return hub.load('SavedModel_model.joblib')

# model = load_model()


# # load model
# savedModel=load_model('gfgModel.h5')
# savedModel.summary()

# save model
tf.saved_model.save(model, 'gfgModel')
# load model
model = tf.saved_model.load('gfgModel')

model_path = 'gs://wagon-data-batch913-drought_detection/MODEL_PATH'


st.markdown("""
    # Drought conditions in Northern Kenya

    Applying deep learning and computer vision for drought resilience, using satellite images and human expert labels to detect drought conditions in Northern Kenya

""")

#**bold** or *italic* text with [links](http://github.com/streamlit) and:
#   - bullet points


st.markdown("""
    Idea for page layout:

        1- what is the problem

        2- whats is the solution

        3-how to solve the problem
""")

#To add original satelite images
st.markdown("""
    ## Satelite images
""")


#image = Image.open('images/kenya_example.png')
#st.image(image, caption='Nothern Kenya', use_column_width=False)


# Import saved model
#somwthing like:
#model = tf.keras.models.load_model("saved_model/satellite.hdf5")


#upload image to be tested
st.header("Image Feature Predictor")
fileUpload = st.file_uploader("Choose a file", type = ['jpg', 'png']) # saves into temporary space



if fileUpload is not None:

    # open user image file
    image_uploaded = Image.open(fileUpload)
    st.image(image_uploaded, caption='Output Image', use_column_width=True)

    # resize image
    image = image_uploaded.resize((65,65))
    st.image(image, caption='Resized Image', use_column_width=True)

    # convert to np.array
    image = np.array(image)
    st.write(image)

    # cast to int64
    image = tf.cast(image, tf.int64) # do we need to make array??

    prediction = model(image)

    # Image labels
    map_dict = {0: "no cows",
                1: "1 cow",
                2: "2 cows",
                3: "3 cows"}

    for m in map_dict.keys():
        if prediction == m:
            print(f'Your region can feed {map_dict[m]}')

    st.markdown("""
        ### Biography

        Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081


    """)

else:
    st.warning('Please upload an Image file :)')
