
import tensorflow as tf
import numpy as np

from keras.layers import Conv2D, Activation, AvgPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential

import pickle, gzip, os
import matplotlib.pyplot as plt

class CNN:

    def __init__(self, train_x, train_y, epochs=100, batch_size=128):

        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y

        # Basic image parameters (aspect ratio is 4:3)
        self.image = {
            "width": 160,
            "height": 120,
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

        # Compile the model with CCE as the loss function and adam as the optimizer 
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
        
        # Print summary of the model
        self.model.summary() 

    def train(self):
        return self.model.fit(self.train_x, self.train_y,
            validation_data=(validation_input, validation_target),
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=1)

    def evaluate(self, test_x, test_y, slack_delta=0.1):
        predictions = self.model.predict(test_x)
        number_of_labels = float(predictions.shape[0])

        correct_predictions = np.less_equal(np.fabs(test_y - predictions), slack_delta)
        accuracy_per_axis = np.sum(correct_predictions, axis=0) / number_of_labels
        accuracy = np.count_nonzero((np.all(correct_predictions, axis=1))) / number_of_labels

        return accuracy, accuracy_per_axis

    def save_model(self, filename):
        self.model.save(filename)

class DataSet:
    def __init__(self, location):
        
        with gzip.open(location, 'rb') as f:
            data_set_dict = pickle.load(f)
            
        self.train_x, self.train_y = data_set_dict['train']['data'], data_set_dict['train']['label']
        self.test_x, self.test_y = data_set_dict['test']['data'], data_set_dict['test']['label']
        
if __name__ == '__main__':
    
    model_filename = 'location_data.h5'
    images_filename = 'location_images.gz'
    
    data_set = DataSet(images_filename)
    
    print("train x shape: ", data_set.train_x.shape)
    print("train y shape: ", data_set.train_y.shape)
    
    cnn = CNN(np.append(data_set.train_x, data_set.test_x, axis=0),
              np.append(data_set.train_y, data_set.test_y, axis=0))
    
    cnn.train()
    
    print("Saving model to " + model_filename)
    
    cnn.save_model(model_filename)
    accuracy = cnn.evaluate(data_set.test_x, data_set.test_y)
    
    print('Accuracy of the model: ' + str(accuracy[0]))
    print('Accuracy per axis(x, y): ' + str(accuracy[1]))
