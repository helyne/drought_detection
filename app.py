import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components




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


image = Image.open('images/kenya_example.png')
st.image(image, caption='Nothern Kenya', use_column_width=False)


# import saved model
#somwthing like:
# model = tf.keras.models.load_model("saved_model/satellite.hdf5")


#upload image to be tested
iploaded_file = st.file_uploader("Choose a image file")

map_dict = {0: "No cows",
            1: "One cow",
            2: "Two cows",
            3: "Three or more cows"
            }





st.markdown("""
    ### Biography

    Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081


""")
