import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from drought_detection.utilities import transform_user_img, make_fig, get_dataframe_data
import pandas as pd
# import cv2
from google.oauth2 import service_account
from google.cloud import storage
import plotly.express as px

################# Streamlit Cloud requirements ################

# Create API client.
#credentials = service_account.Credentials.from_service_account_info(
#    st.secrets["gcp_service_account"]
#)
#client = storage.Client(credentials=credentials)

#bucket_name = "wagon-data-batch913-drought_detection"
#file_path = "SavedModel/Model_3band_RGB_ha/"

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
#@st.experimental_memo(ttl=1200)
#def read_file(bucket_name, file_path):
#    bucket = client.bucket(bucket_name)
 #   content = bucket.blob(file_path).download_as_string().decode("utf-8")
 #   return content

#content = read_file(bucket_name, file_path)


################# MODEL ##################

# Load the model from the cloud
STORAGE_LOCATION = f'gs://wagon-data-batch913-drought_detection/SavedModel/Model_3band_RGB' # GCP path


# load model (cache so it only loads once and saves time)
@st.cache
def load_model(STORAGE_LOCATION):
    return tf.saved_model.load(STORAGE_LOCATION)

model = load_model(STORAGE_LOCATION)

################# WEBSITE #################

st.markdown("""
    # Drought conditions in Northern Kenya
    Applying deep learning and computer vision for drought resilience, using satellite images and human expert labels to detect drought conditions in Northern Kenya
""")

#**bold** or *italic* text with [links](http://github.com/streamlit) and:
#   - bullet points

tab1, tab2, tab3 = st.tabs(["Home", "Background", "Meet the team"])

with tab1:
    st.header("Image Feature Predictor")

    #upload image to be tested
    fileUpload = st.file_uploader("Choose a file", type = ['jpg', 'png']) # saves into temporary space

    if fileUpload is not None:
        # open user image file
        image_uploaded = Image.open(fileUpload)

        #shows uploaded image to the web
        st.title("Here is the image you've selected")
        st.image(image_uploaded)

        # transform user image
        image = transform_user_img(image_uploaded)

        # create cached function to predict from image
        @st.cache # cache so only loads once if the input is the same (saves reloading time)
        def predict(image):
            #predict from web:
            with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], STORAGE_LOCATION)
                graph = tf.compat.v1.get_default_graph()
                prediction = sess.run('StatefulPartitionedCall:0',
                                      feed_dict={'serving_default_keras_layer_input:0': image})
            return prediction

        # use predict function
        prediction = predict(image)
        result = max(prediction[0])

        st.header('Drought Prediction')
        if result == prediction[0][0]:
            st.error("Drought :desert: :sweat: :sunny:  : Looks like your region is likely suffering \
                from drought and there's not enough plant life in the area to feed any cows.")
        elif result == prediction[0][1]:
            st.warning("Drought risk :cow: :warning:  Looks like your region is at risk of drought. \
                The forage quality can only feed one cow per 20sqm area.")
        elif result == prediction[0][2]:
            st.info("Possible drought risk :cow: :seedling: :cow: : Looks like your region is not \
                suffering from a drought, however, it can likely only feed 2 cows per 20sqm area.")
        elif result == prediction[0][3]:
            st.success("No Drought :herb: :cow: :cow: :cow: :deciduous_tree:  : Looks like your region \
                is healthy and can feed at least 3 cows per 20sqm area! Happy foraging!")


        st.title("Here's the likelihood of your region being in drought")

        # get dataframe
        df = get_dataframe_data(prediction)

        # set variables and make plot
        x = df.iloc[:,2]
        y = df.iloc[:,1]
        fig = make_fig(df, x, y)

        # display plot
        st.plotly_chart(fig)




with tab2:
    st.header("Background")

    st.write("**Problem/Opportunity**")
    st.write("When it comes to droughts, Index insurance companies rely on models to predict average losses in a region and provide farmers with cash when losses reach a certain threshold.")
    st.write("Can these models be improved upon to better predict drought conditions, with the end goal being a more effective cash transfer to the impacted farmers ?")
    image1 = Image.open('images/green.jpg')
    st.image(image1)


    st.write("**Solution**")
    st.write("We propose an improved version of an existing model, increasing its drought prediction accuracy by 155%.")

    st.subheader('What is our model based on?')
    st.write("**Efficientnet Architecture**")
    image2 = Image.open('images/EfficientNet.png')
    st.image(image2)
    st.write("A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.")
    st.write("In our project of Drought Detection, we have use CNN based on an EfficientNet algorithm. EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. It has been shown that in terms of accuracy it is the most optimal and also, it does not take as much parameters as other algorithms such as ResNet, SENet, NASNet-A  and AmoebaNet-C.")



    st.header("Bibliography")
    st.write("Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081")



with tab3:
    col1, col2 = st.columns([1, 3])
    col1.image('images/Helyne.png', use_column_width=True)
    col2.write("**Helyne Adamson**")
    col2.write("I'm a cognitive science researcher and general neuroscience enthusiast. In my recent work, I used diffusion MRI methods to analyze neuroanatomical brain changes during foreign language learning in adults.")
    col2.write("*Linkedin page*: https://www.linkedin.com/in/helyne-adamson-1587939/?originalSubdomain=de")
    col1, col2 = st.columns([1, 3])
    col1.image('images/Lluis.png', use_column_width=True)
    col2.write("**Lluis Morey**")
    col2.write("Hello, my name is Lluis Morey and I am 22 years old. I am from Spain, more particularly from the island of Mallorca. I am currently in my third year of Business Management. Since I started at university I have been very interested in the world of technology. Data Science is deeply connected to decision making at a business level, this is what caught my interest. I would love to work as a data scientist after Le Wagon.")
    col2.write("*Linkedin page*: https://www.linkedin.com/in/lluismorey/")
    col1, col2 = st.columns([1, 3])
    col1.image('images/Marie.png', use_column_width=True)
    col2.write("**Marie-Laure Geai**")
    col2.markdown("I am a Geographic Information Systems analyst (cartographer): I work with spatial data. My niche area of interest is 3D mapping of alpine areas using various imagery sources for hazard monitoring (or ideally, for outdoor practices!)")
    col2.write("*Linkedin page*: https://www.linkedin.com/in/mariegeai/")
    col1, col2 = st.columns([1, 3])
    col1.image('images/Patricia.png', use_column_width=True)
    col2.write("**Patricia Regina Soares de Souza**")
    col2.write("I am a Senior Research Scientist with a doctoral degree in Immunopharmacology. I felt in love with data science and machine learning during a recent project where I identified serum biomarkers that could predict the responssiveness of treatment in patients with Rheumathoid Arthritis. And so, I have decided to learn more about big data and become a Data Scientist.")
    col2.write("*Linkedin page*: https://www.linkedin.com/in/patricia-regina-soares-de-souza-2b7a7042/")
