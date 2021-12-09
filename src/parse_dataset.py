import numpy as np
import pickle, os, gzip

from PIL import Image
from sklearn.model_selection import train_test_split

path_to_images = "../data/images/"
dataset_filename = "../data/dataset_pickled.gz"

# Extract label information from images 
def extract_data(images):
    x = []
    y = []

    for image_name in images:
        path = path_to_images + image_name

        np_array_of_image = np.asarray(Image.open(path).convert("RGB"), dtype="float64") / 255
        x.append(np_array_of_image)
        
        label = [float(c) for c in image_name[:image_name.index(".jpg")].split(",")]
        y.append(np.asarray(label))

    return x, y

# Get all images from data director
all_images = os.listdir(path_to_images)
print("Extracting images from " + path_to_images)

# Amount of images for training and testing
test_size = 0.1
test_amount = int(len(all_images)* test_size)

# Split images into seperate lists for training and testing
train_data_images = all_images[test_amount:]
test_data_images = all_images[:test_amount]

# Extract labels from training and testing images
train_x, train_y = extract_data(train_data_images)
test_x, test_y = extract_data(test_data_images)

# Create dictionary containg dataset information
data = dict()

# Add training information
data["train"] = {}
data["train"]["data"] = np.array(train_x)
data["train"]["label"] = np.array(train_y)

# Add testing information
data["test"] = {}
data["test"]["data"] = np.array(test_x)
data["test"]["label"] = np.array(test_y)

# Dump dataset to file
print("Dumping dataset to " + dataset_filename)

gzip_file = gzip.open(dataset_filename, "wb")
pickle.dump(data, gzip_file)
gzip_file.close()

print("Finished.")
