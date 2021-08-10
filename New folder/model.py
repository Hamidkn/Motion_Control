import keras.models
# import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Model():
    MAX_EPOCHS = 10
    multi_val_performance = {}
    multi_performance = {}
    # num_label=2
    # hiddennodes=40
    # OUT_STEPS = 1

    def generate_model(self):
        
        model = Sequential()
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse'])
        return model

    def compile(self, model, lr=0.001):
        model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse'])

    def fit(self, model, XTrain, YTrain, MAX_EPOCHS):
            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            #                                         patience=patience,
            #                                         mode='min')
            # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            self.logger.info("Training is started ... ")
            history = model.fit(XTrain, YTrain, epochs=MAX_EPOCHS,
                    #  validation_data=window.val,
                  #    callbacks=[early_stopping,
                   #  callback
                         #       ]
                     )
            return history

    def summary(self, model):
        with open('model_summary.txt','w') as file:
            with redirect_stdout(file):
                model.summary()
    