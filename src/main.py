
import tensorflow as tf
import numpy as np

from keras.layers import Conv2D, Activation, AvgPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential

import pickle, os
import matplotlib.pyplot as plt

class CNN:

    def __init__(self, train_x, train_y, epochs=100, batch_size=128):

        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y

        # Basic image parameters
        self.image = {
            "width": 6,
            "height": 9,
            "depth": 1
        }

        # Create sequential CNN model 
        self.model = Sequential([

            Conv2D(32, 3, activation="relu", input_shape=(self.image["width"], self.image["height"], self.image["depth"])),
            Activation("relu"),
            AvgPool2D(pool_size=(2, 2)),

            Conv2D(32, 3, activation="relu"),
            Activation("relu"),
            AvgPool2D(pool_size=(2, 2)),

            Conv2D(16, 3),
            Activation("relu"),

            Conv2D(16, 3),
            Activation("relu"),
            AvgPool2D(pool_size=(2, 2)),

            Conv2D(16, 3),
            Activation("relu"),

            Conv2D(8, 3),
            Activation("relu"),

            Flatten(),
            Dense(2)
        ])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

    def train(self):
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epochs)

    def evaluate(self, test_x, test_y, slack_delta=0.1):
        predictions = self.model.predict(test_x)
        number_of_labels = float(predictions.shape[0])

        correct_predictions = np.less_equal(np.fabs(test_y - predictions), slack_delta)
        accuracy_per_axis = np.sum(correct_predictions, axis=0) / number_of_labels
        accuracy = np.count_nonzero((np.all(correct_predictions, axis=1))) / number_of_labels

        return accuracy, accuracy_per_axis

    def save_model(self, filename):
        self.model.save(filename)


# Amount of data to train on
testing_percent = 0.7



data_dir = pathlib.Path(dataset_path)

data = []

# Get the number of images to use
training_amount = int(len(data) * testing_percent)

# Load in the training and testing data
train_input, validation_target = data[:training_amount], data[training_amount:]
train_target, validation_input = data[:training_amount], data[training_amount:]

# Normalizing data - turn values 0-255 into 0-1
train_input = train_input / 255.0
validation_target = validation_target / 255.0

# Turn array of integers into a binary class matrix
y_train_one_hot = to_categorical(train_target)
y_test_one_hot = to_categorical(validation_input)

# Create a new model for the dataset with optimised layers - in this case, not so much
model = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(110, 110, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),

    Dense(8, activation="softmax")
])

# Categorial crossentropy can also be used
# "categorical_crossentropy"

# Use MSE as the loss function
model.compile(loss=tf.keras.losses.MeanSquaredError(), 
              optimizer="adam",
              metrics=["accuracy"])

trained_model = model.fit(train_input, train_target,
                    validation_data=(validation_input, validation_target),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    verbose=1)


print(len(x_train))
