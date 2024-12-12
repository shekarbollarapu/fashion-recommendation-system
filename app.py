from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
import joblib
from tqdm import tqdm

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)
    return result_normlized

# List of image paths
img_files = []

# Assuming the fashion images are in the 'fashion_small/images' directory
for fashion_images in os.listdir('fashion_small/images'):
    images_path = os.path.join('fashion_small/images', fashion_images)
    img_files.append(images_path)

# Extracting image features
image_features = []

for files in tqdm(img_files):
    features_list = extract_features(files, model)
    image_features.append(features_list)

# Save the features and file list using joblib
joblib.dump(image_features, 'image_features_embedding.joblib')
joblib.dump(img_files, 'img_files.joblib')

print("Feature extraction and saving completed!")
