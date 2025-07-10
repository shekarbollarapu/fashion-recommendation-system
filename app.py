import streamlit as st
import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
import os

# Load saved features and image paths from pickle files
@st.cache_data
def load_features():
    with open("image_features_embedding.pkl", "rb") as f:
        features = pickle.load(f)
    with open("img_files.pkl", "rb") as f:
        img_files = pickle.load(f)
    return features, img_files

features_list, img_files = load_features()

# Build model for extracting features (if needed to extract from user uploaded image)
@st.cache_resource
def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = build_model()

# Function to extract features from an image path
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    features = model.predict(preprocessed_img).flatten()
    features_norm = features / norm(features)
    return features_norm

# Function to recommend similar images
def recommend(image_feature, features_list, img_files, top_k=5):
    similarities = []
    for feat in features_list:
        sim = np.dot(image_feature, feat)
        similarities.append(sim)
    # Get indices of top_k similar images
    indices = np.argsort(similarities)[::-1][1:top_k+1]  # skip first as it's the same image
    recommended_images = [img_files[i] for i in indices]
    return recommended_images

# Streamlit app UI
st.title("Fashion Recommendation System")

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show uploaded image
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)
    
    # Extract features from uploaded image
    uploaded_img_features = extract_features(temp_path, model)
    
    # Recommend similar images
    recommended_imgs = recommend(uploaded_img_features, features_list, img_files, top_k=5)
    
    st.write("### Recommended similar fashion items:")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        img_path = recommended_imgs[idx]
        if os.path.exists(img_path):
            col.image(img_path, use_column_width=True)
        else:
            col.write(f"Image not found: {img_path}")
    
    # Remove temporary uploaded image
    os.remove(temp_path)

else:
    st.write("Please upload an image to get recommendations.")
