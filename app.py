import streamlit as st
from PIL import Image
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




st.markdown("""
    ### Biography

    Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya. https://doi.org/10.48550/arXiv.2004.04081


""")
