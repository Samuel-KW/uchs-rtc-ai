import tensorflow as tf
import numpy as np

from keras.models import load_model
from PIL import Image

path_to_images = "../data/images/"
dataset_filename = "../data/dataset_pickled.gz"
model_filename = "../data/model.h5"

model = load_model(model_filename)

def predict(image):
  
    test_x = np.asarray(Image.open(test_image_path).convert("L"), dtype="float64") / 255
    test_x_resized = test_x.reshape(1, 150, 150, 1)

    test_y = model.predict(test_x_resized)
    flattened_test_y = test_y.flatten()
    
    x = flattened_test_y[0]
    y = flattened_test_y[1]
    
    print('Sending data: ' + test_y_to_vicon)
