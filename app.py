import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from drought_detection.gcp import storage_upload
import io
import tempfile
import pandas as pd


st.markdown("""
    # Drought conditions in Northern Kenya

    Applying deep learning and computer vision for drought resilience, using satellite images and human expert labels to detect drought conditions in Northern Kenya

""")

#**bold** or *italic* text with [links](http://github.com/streamlit) and:
#   - bullet points




tab1, tab2, tab3 = st.tabs(["Home", "Under the hood", "Meet the team"])

with tab1:
    st.header("Image Feature Predictor")

    def saveImage(byteImage):
        """The Streamlit file picker converts the image file to a byte stream so we need to
        convert that byte stream into an Image using io.BytesIO"""
        bytesImg = io.BytesIO(byteImage)
        imgFile = Image.open(bytesImg)

        return imgFile, bytesImg

    # Uploads an image to be tested

    fileUpload = st.file_uploader("Choose a file", type = ['jpg', 'png'])


    # Process the uploaded file using the code in the XXX.py file
    if fileUpload:
        file = fileUpload.read()
        path, bytesImg = saveImage(file)
        st.write(type(path))
        st.write(type(bytesImg))
        st.image(path, width = 300) #, height = 100)
        model = load_model()
        # imgFs = predictFeatures(path, model)
        # tableArea = st.empty()
        # tableArea = tableArea.table()
        # with st.spinner("Prediction In Progress"):
        #     for ix, (key, val) in enumerate(imgFs):
        #         tableArea.add_rows(pd.DataFrame({"Attribute": key, "Percentage %": val}, index = [ix]))
        #     st.success("Success")



    # #upload image to be tested
    # uploaded_file = st.file_uploader("Choose a file", type = ['jpg', 'png'])

    # map_dict = {0: "no cows",
    #             1: "1 cow",
    #             2: "2 cows",
    #             3: "3 cows"}




    # if uploaded_file is not None:
    #     def saveImage(uploaded_file):
    #         bytesImg = io.BytesIO(uploaded_file)
    #         imgFile = Image.open(bytesImg)
    #         return bytesImg

    #     # get user image
    #     user_img = saveImage(uploaded_file)
    #     # resize image
    #     user_img = user_img.resize((65,65))
    #     # get all testing images as NumPy array
    #     user_img = np.array(user_img)

    #     #Needs a script to convert the uploaded image in the right format
    #     formated_image = uploaded_file


    #     #prepro

    #     #shows uploaded image to the web
    #     st.image(formated_image)

    #     # Generate prediction



    #     Genrate_pred = st.button("Generate Prediction")
    # #    if Genrate_pred:
    # #        # Loads saved model
    # #        saved_model=load_model('model.h5') #check path

    # #        prediction = model.predict()
    # #        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))

    st.header("Biography")
    st.write("Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081")


with tab2:
    st.header("Under the hood")
    st.write("**Problem/Opportunity**")
    st.write("When it comes to droughts, Index insurance companies rely on models to predict average losses in a region and provide farmers with cash when losses reach a certain threshold.")
    st.write("Can these models be improved upon to better predict drought conditions, with the end goal being a more effective cash transfer to the impacted farmers ?")
    image1 = Image.open('images/green.jpg')
    st.image(image1)
    \
    st.write("**Solution**")
    st.write("We propose an improved version of an existing model, increasing its drought prediction accuracy by x%. We leverage our model in an app [...]")

    st.subheader('What is our model based on?')
    st.write("**Efficientnet Architecture**")
    image2 = Image.open('images/EfficientNet.png')
    st.image(image2)
    st.write("A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.")
    st.write("In our project of Drought Detection, we have use CNN based on an EfficientNet algorithm. EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. It has been shown that in terms of accuracy it is the most optimal and also, it does not take as much parameters as other algorithms such as ResNet, SENet, NASNet-A  and AmoebaNet-C.")




with tab3:
    columns = st.columns(2)
    image = Image.open('images/Helyne.png')
    columns[0].image(image, use_column_width=True)
    columns[1].write("**Helyne Adamson**")
    columns[1].write("I'm a cognitive science researcher looking to transition into industry. My dissertation applied diffusion MRI methods to analyze neuroanatomical brain changes during foreign language learning in adults. While I have some experience with the command line and R, I've wanted to learn Python and more advanced analysis methods (machine learning & deep learning) for a while now. I'm ultimately aiming for a data position in green/education/medical tech.")
    columns[1].write("*Linkedin page*: https://www.linkedin.com/in/helyne-adamson-1587939/?originalSubdomain=de")

    columns = st.columns(2)
    image = Image.open('images/Lluis.png')
    columns[0].image(image, use_column_width=True)
    columns[1].write("**Lluis Morey**")
    columns[1].write("Hello, my name is Lluis Morey and I am 22 years old. I am from Spain, more particularly from the island of Mallorca. I am currently in my third year of Business Management. Since I started at university I have been very interested in the world of technology. Data Science is deeply connected to decision making at a business level, this is what caught my interest. I would love to work as a data scientist after Le Wagon.")
    columns[1].write("*Linkedin page*: https://www.linkedin.com/in/lluismorey/")

    columns = st.columns(2)
    image = Image.open('images/Marie.png')
    columns[0].image(image, use_column_width=True)
    columns[1].write("**Marie-Laure Geai**")
    columns[1].write("""I am a Geographic Information Systems analyst (cartographer): I work with spatial data. I have worked for the last 8 years in Paris, in the energy industry. My tasks include cross-referencing location-based parameters to identify optimal areas to install infrastructure. I used to work in glaciology research labs in the US and Canada, using aerial and satellite imagery. My niche area of interest is 3D mapping of alpine areas using various imagery sources for hazard monitoring (or ideally, for outdoor practices! I love mountaineering)""")
    columns[1].write("*Linkedin page*: https://www.linkedin.com/in/mariegeai/")

    columns = st.columns(2)
    image = Image.open('images/Patricia.png')
    columns[0].image(image, use_column_width=True)
    columns[1].write("**Patricia Regina Soares de Souza**")
    columns[1].write("I am a Senior Research Scientist with a doctoral degree in Immunopharmacology. After years working in a lab generating data, I have become interested in working with the data I have generated in a more powerful way.")
    columns[1].write("*Linkedin page*: https://www.linkedin.com/in/patricia-regina-soares-de-souza-2b7a7042/")
