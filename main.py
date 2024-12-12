import streamlit as st
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image

# Load precomputed features and image paths using joblib
try:
    features_list = joblib.load("image_features_embedding.joblib")
    img_files_list = joblib.load("img_files.joblib")
except Exception as e:
    st.error(f"Error loading the files: {e}")
    raise

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Streamlit app title
st.title('Fashion Recommender System')

# Function to save uploaded file
def save_file(uploaded_file):
    try:
        if not os.path.exists("uploader"):
            os.makedirs("uploader")
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

# Function to extract features from uploaded image
def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)
    return result_normlized

# Function to recommend similar images based on features
def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Upload image functionality
uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        # Display uploaded image
        show_images = Image.open(uploaded_file)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)
        
        # Extract features of the uploaded image
        features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
        
        # Get recommended images
        img_indicess = recommend(features, features_list)
        
        # Display recommended images
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.header("I")
            st.image(img_files_list[img_indicess[0][0]])

        with col2:
            st.header("II")
            st.image(img_files_list[img_indicess[0][1]])

        with col3:
            st.header("III")
            st.image(img_files_list[img_indicess[0][2]])

        with col4:
            st.header("IV")
            st.image(img_files_list[img_indicess[0][3]])

        with col5:
            st.header("V")
            st.image(img_files_list[img_indicess[0][4]])

    else:
        st.header("An error occurred while uploading the file.")
