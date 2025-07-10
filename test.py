import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

# ✅ Correct file loading
features_list = np.load("image_features_embedding.pkl", allow_pickle=True)
img_files_list = pickle.load(open("img_files.pkl", "rb"))

print(np.array(features_list).shape)

# Load feature extraction model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Load and preprocess query image
img = image.load_img('sample/shoes.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expand_img = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expand_img)

# Extract features and normalize
result_to_resnet = model.predict(preprocessed_img)
flatten_result = result_to_resnet.flatten()
result_normlized = flatten_result / norm(flatten_result)

# Fit Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features_list)

# Get recommendations
distance, indices = neighbors.kneighbors([result_normlized])

print("Recommended indices:", indices)

# Display recommended images (excluding the first as it's the input image)
for file_index in indices[0][1:6]:
    print("Recommended image:", img_files_list[file_index])
    tmp_img = cv2.imread(img_files_list[file_index])
    tmp_img = cv2.resize(tmp_img, (200, 200))
    cv2.imshow("Recommended", tmp_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
