import numpy as np
from PIL import Image
from keras.applications.xception import Xception
from keras.utils import get_file
from tqdm import tqdm
import os

def get_xception_model():
    weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_path = get_file("xception_weights.h5", weights_url)
    return Xception(include_top=False, pooling='avg', weights=weights_path)

def extract_features(directory, model):
    features = {}
    valid_images = ['.jpg', '.jpeg', '.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = os.path.join(directory, img)
        image = Image.open(filename).resize((299,299))
        image = np.expand_dims(np.array(image)/127.5 - 1.0, axis=0)
        features[img] = model.predict(image)
    return features
