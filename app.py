import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from drought_detection.gcp import storage_upload




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
uploaded_file = st.file_uploader("Choose a image file")

map_dict = {0: "Annual Crop",
            1: "Forest",
            2: "Herbaceous Vegetation",
            3: "Highway",
            4: "Industrial",
            5: "Pasture",
            6: "Permanent Crop",
            7: "Residential",
            8: "River",
            9: "Sea or Lake"
            }

if uploaded_file is not None:
    # Needs a script to convert the uploaded image in the right format
    #formated_image = image
    #shows uploaded image to the web
    #st.image(formated_image)

    # Generate prediction
    Genrate_pred = st.button("Generate Prediction")
#    if Genrate_pred:
#        # Loads saved model
#        saved_model=load_model('model.h5') #check path

#        prediction = model.predict()
#        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))




st.markdown("""
    ### Biography

    Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081


""")
