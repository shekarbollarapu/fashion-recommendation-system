import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
import joblib
import os

# Load precomputed features
features_list = joblib.load("image_features_embedding.joblib")
img_files_list = joblib.load("img_files.joblib")

# Initialize the model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

def save_file(uploaded_file):
    """Save the uploaded file to 'uploader' directory."""
    upload_dir = "uploader"
    os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists
    try:
        with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join(upload_dir, uploaded_file.name)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_img_features(img_path, model):
    """Extract features from an uploaded image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    return flatten_result / norm(flatten_result)

def recommend(features, features_list):
    """Find similar images using nearest neighbors."""
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Upload and process an image
uploaded_file = st.file_uploader("Choose an image to find similar fashion items:")
if uploaded_file:
    img_path = save_file(uploaded_file)
    if img_path:
        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        features = extract_img_features(img_path, model)
        indices = recommend(features, features_list)

        # Display recommendations
        st.subheader("Recommended Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices[0]):
                col.image(img_files_list[indices[0][i]], use_column_width=True)
