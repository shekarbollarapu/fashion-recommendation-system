import streamlit as st
import joblib
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Streamlit app title
st.title('Fashion Recommender System')

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Load precomputed features and image paths using joblib
try:
    features_list = joblib.load("image_features_embedding.joblib")
    img_files_list = joblib.load("img_files.joblib")
    st.success("Successfully loaded the precomputed features!")
except Exception as e:
    st.error(f"Error loading the files: {e}")
    raise

# Function to save uploaded file
def save_file(uploaded_file):
    try:
        upload_dir = "uploader"
        os.makedirs(upload_dir, exist_ok=True)
        with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join(upload_dir, uploaded_file.name)
    except Exception as e:
        st.error(f"Error saving the file: {e}")
        return None

# Function to extract features from uploaded image
def extract_img_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normlized = flatten_result / norm(flatten_result)
        return result_normlized
    except Exception as e:
        st.error(f"Error extracting features from image: {e}")
        return None

# Function to recommend similar images based on features
def recommend(features, features_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        distence, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error finding neighbors: {e}")
        return []

# Upload image functionality
uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    img_path = save_file(uploaded_file)
    if img_path:
        # Display uploaded image
        show_images = Image.open(img_path)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)

        # Extract features of the uploaded image
        features = extract_img_features(img_path, model)
        if features is not None:
            # Get recommended images
            img_indicess = recommend(features, features_list)

            # Display recommended images
            col1, col2, col3, col4, col5 = st.columns(5)

            for idx, col in enumerate([col1, col2, col3, col4, col5]):
                if idx < len(img_indicess[0]):
                    with col:
                        st.header(f"Recommendation {idx + 1}")
                        st.image(img_files_list[img_indicess[0][idx]])

    else:
        st.header("An error occurred while uploading the file.")
