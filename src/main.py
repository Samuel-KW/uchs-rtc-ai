
import tensorflow as tf
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, AvgPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential

import pickle, gzip
import matplotlib.pyplot as plt

# https://mathworld.wolfram.com/TukeysBiweight.html
def tukey_bi_weight_loss(y_true, y_predicted):
        z = y_true - y_predicted
        z_abs = tf.abs(z)
        c = 4.685
        subset_bool = tf.less_equal(z_abs, c)
        subset = tf.cast(subset_bool, z_abs.dtype)
        inv_subset = tf.cast(tf.logical_not(subset_bool), z_abs.dtype)
        c_sq_by_six = c ** 2 / 6
        return (1 - ((1 - ((z / c) ** 2)) ** 3) * subset + inv_subset) * c_sq_by_six

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

            Conv2D(32, 3, activation="relu", input_shape=(self.image["height"], self.image["width"], self.image["depth"])),
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
            loss="MAE",
            optimizer="adam",
            metrics=["accuracy"])
        
        # Print summary of the model
        #self.model.summary() 

    def train(self, test_x, test_y):

        print("train x", self.train_x.shape)
        print("train y", self.train_y.shape)

        return self.model.fit(self.train_x, self.train_y,
            validation_data=(test_x, test_y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True)

    def evaluate(self, test_x, test_y, slack_delta=0.1):
        predictions = self.model.predict(test_x)
        number_of_labels = float(predictions.shape[0])

        print("predictions", predictions)

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
    
    model_filename = '../data/model.h5'
    images_filename = '../data/dataset_pickled.gz'
    
    data_set = DataSet(images_filename)
    
    cnn = CNN(data_set.train_x, data_set.train_y)
    
    trained_model = cnn.train(data_set.test_x, data_set.test_y)
    

    print("Saving model to " + model_filename)
    
    cnn.save_model(model_filename)
    accuracy = cnn.evaluate(data_set.test_x, data_set.test_y)
    
    print('Accuracy of the model: ' + str(accuracy[0]))
    print('Accuracy per axis(x, y): ' + str(accuracy[1]))

    
    plt.figure(1)
    plt.plot(trained_model.history['acc'])
    plt.plot(trained_model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
