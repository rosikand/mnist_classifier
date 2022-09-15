"""
File: app.py
------------------
Simple data app. 
"""

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import models
import time


def load_image(image_file):
	img = Image.open(image_file)
	return img

def inference_function(img):
    net = models.ModelInterface()
    net.load_weights("saved/trained_weights.pth")
    pred = net.predict(img)
    return pred

st.write("""
# Binary MNIST Classifier App 

This app predicts whether a digit is a 0 or a 1. Give it a try! 
""")

st.sidebar.subheader('User Input')

# selected_model = st.sidebar.selectbox(
#     'Choose a model',
#     ('MLP',))

# st.sidebar.write('You selected:', selected_model)

image_file = st.sidebar.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if image_file is not None:
    img = load_image(image_file)
    img_array = np.array(img)
    st.write("#### Your image:")
    st.image(img_array)
    
    st.write('#### Classification:')
    
    with st.spinner('Model predicting...'):
        time.sleep(1)
        pred = inference_function(img_array)
        st.write("Prediction: ", pred)
