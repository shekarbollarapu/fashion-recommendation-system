from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import joblib  # Updated from pickle

# Initialize model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    """Extract feature embeddings for an image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    return flatten_result / norm(flatten_result)

# Collect all image paths
image_dir = 'fashion_small/images'
assert os.path.exists(image_dir), f"Directory {image_dir} does not exist."
img_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

# Extract image features
image_features = []
for img_path in tqdm(img_files, desc="Extracting image features"):
    image_features.append(extract_features(img_path, model))

# Save features and file paths
joblib.dump(image_features, "image_features_embedding.joblib")
joblib.dump(img_files, "img_files.joblib")
print("Feature extraction completed.")
