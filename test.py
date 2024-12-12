import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

# Load precomputed features
features_list = joblib.load("image_features_embedding.joblib")
img_files_list = joblib.load("img_files.joblib")

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Test image path
img = image.load_img('sample/shoes.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expand_img = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expand_img)
result_to_resnet = model.predict(preprocessed_img)
flatten_result = result_to_resnet.flatten()
result_normlized = flatten_result / norm(flatten_result)

# Find similar items
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features_list)
distances, indices = neighbors.kneighbors([result_normlized])

print("Recommended Items:")
for file in indices[0][1:]:
    print(img_files_list[file])
    tmp_img = cv2.imread(img_files_list[file])
    tmp_img = cv2.resize(tmp_img, (200, 200))
    cv2.imshow("Recommendation", tmp_img)
    cv2.waitKey(0)
