import numpy as np
import pandas as pd
import tensorflow as tf
# from google.cloud import storage
# imports from our parameters file
from drought_detection.params import GCP_BUCKET_NAME, MODEL_NAME
import plotly.express as px



############################### Webapp functions ##################################


def transform_user_img(image_uploaded, new_height, new_width):
    # convert PIL image to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image_uploaded)

    # remove 4th channel if it exists
    # In case of grayScale images the len(img.shape) == 2
    if len(image.shape) > 2 and image.shape[2] == 4:
        # slice off the alpha channel
        image = image[:, :, :3]

    # resize image
    image = tf.image.resize(image, [new_height, new_width])

    # create batch axis
    image = tf.expand_dims(image, 0)

    return image

def get_dataframe_data(prediction):
    df = pd.DataFrame(prediction.transpose())
    df[0] = (df[0] * 100)
    df[0] = df[0].astype(int)
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'Classification', 0:f"% Confidence"})
    df['Number of cows'] = ['0 cows', '1 cow', '2 cows', '3 cows or more']
    df['Percent'] = df[f"% Confidence"].astype(str)
    df['Percent'] = df['Percent'].map(lambda x: x + ' %')
    df['Drought Severity'] = ['Drought: Very Severe', 'Drought: Severe', 'Drought: Likely', 'Drought: Unlikely']
    return df

def make_fig(df, x, y):
    fig = px.scatter(df, x, y,
                        size=f"% Confidence", color='Number of cows',
                        # color_discrete_sequence=px.colors.qualitative.Vivid,
                        color_discrete_sequence=['#FF0D0D', '#E95302', '#FFBF00', '#238823'],
                        hover_name='Drought Severity', size_max=50)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(range=[-10, 119])
    full_fig = fig.update_layout(
        title={
        'text': "Local Drought Assessment",
        'x':0.47,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title="Number of Cows",
        yaxis_title="% Confidence",
        legend_title="Forage Quality",
        plot_bgcolor='#FFFFFF',
        font=dict(
            family="Open Sans, verdana, arial, sans-serif",
            size=15,
            color="#000000"))
    return full_fig


############################### Save functions ##################################


def save_model(model, MODEL_NAME):
    # set storage location in cloud
    STORAGE_LOCATION = f'gs://wagon-data-batch913-drought_detection/SavedModel/{MODEL_NAME}'
    # save directly to gcp
    model.save(STORAGE_LOCATION)
    print(f"uploaded model to gcp cloud storage under \n => {STORAGE_LOCATION}")
    print("===========================saved model========================")

# exporting metrics locally
def export_to_csv(metric_dict):
    '''Export accuracy, val_acc and test_acc of a model into a csv file'''
    df = pd.DataFrame(metric_dict)
    history_file = f"{MODEL_NAME}_accuracy.csv"
    df.to_csv(history_file)
    return history_file

# # saving metrics to GCP
# def save_csv(history_file):
#     BUCKET_NAME='wagon-data-batch913-drought_detection'
#     STORAGE_LOCATION = f'SavedModel/{MODEL_NAME}/{MODEL_NAME}_accuracy.csv'

#     client = storage.Client().bucket(BUCKET_NAME)
#     blob = client.blob(STORAGE_LOCATION)
#     blob.upload_from_filename(history_file)
#     print(f"uploaded model accuracy to gcp cloud storage under \n => {STORAGE_LOCATION}")
