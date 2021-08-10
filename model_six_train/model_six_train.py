import csv
import keras.models
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler
import math, sklearn.metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.linear_model import LinearRegression
import pickle


initial_learning_rate = 0.01
epochs = 300
decay = initial_learning_rate / epochs
def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)
#     # drop_rate = 0.5
#     # epochs_drop = 10.0
#     # return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
#     return lr * 1 / (1 + decay * epoch)

# def normalize(arr, t_min=0, t_max=1):
#     norm_arr = []
#     diff = t_max - t_min
#     diff_arr = max(arr) - min(arr)    
#     for i in arr:
#         temp = (((i - min(arr))*diff)/diff_arr) + t_min
#         norm_arr.append(temp)
#     return norm_arr

dataframe = 'E:/UniversityFiles/Control-20210210T123743Z-001/Control/homework/MainModel_Version1/Mainmodel_vnew/Motion_Control/model_six_train/dataframe1/dataframe_main_model.txt'

df = pd.read_csv(dataframe)
print(df)
window = 15
# dataset = df.values
X = df[['Ftotal','velocity']].values
Y = df[['position']].values
# print(dataset)

# X = dataset[0:54999,:]
# Y = dataset[55001:110000,:]


min_max_scaler = preprocessing.MinMaxScaler()
s_x = MinMaxScaler()
X = s_x.fit_transform(X)
s_y = MinMaxScaler()
Y = s_y.fit_transform(Y)
# print(X_scale)

# X = X_scale[:,2:3]
# Y = X_scale[:,-1]
# print(X)
# print(Y)
# min_max_scaler = preprocessing.MinMaxScaler()
# Y = min_max_scaler.fit_transform(Y.reshape(-1,1))

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.25)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential()
model.add(Dense(500, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))

model.compile(loss=tf.keras.losses.MeanSquaredError(), 
        optimizer =RMSprop(learning_rate=initial_learning_rate), 
        metrics=['mse'])
start = time.time()
# model_info = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_data=(X_val, Y_val))
model_info = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_data=(X_val, Y_val))
                # callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)])
stop = time.time()
print(f'Training time: {stop - start} s')
plot_model(model, to_file='model_six_train/Images/model_six_RMSPROP.png', show_shapes=True, rankdir="LR")
with open('model_six_train/summarymodel_model_six_RMSPROP.txt','w') as file:
            with redirect_stdout(file):
                model.summary()
model.save('model_six_train/model/model_6.h5')

loss, mse = model.evaluate(X_test, Y_test)
print(f'mse: {mse}, loss: {loss}')

y_pred = model.predict(X_test)

# Unscale data
# Xtest_us = s_x.inverse_transform(X_test[:,-1,:])
ytest_us = s_y.inverse_transform(Y_test)
yp = s_y.inverse_transform(y_pred)

regressor = LinearRegression()
regressor.fit(ytest_us, y_pred)
y_fit = regressor.predict(y_pred)

plt.plot(model_info.history['mse'])
plt.plot(model_info.history['val_mse'])
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train','Val'],loc='upper right')
plt.savefig('model_six_train/Images/fig_Accuracy_Rmsprop_mse.png')
plt.show()
       
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'],loc='upper right')
plt.savefig('model_six_train/Images/fig_loss_Rmsprop_mse.png')
plt.show()

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

# plt.scatter(ytest_us, yp, color='blue', label= 'data')
# plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
# plt.title('Linear Regression')
# plt.legend()
# plt.xlabel('observed')
# plt.ylabel('predicted')
# plt.savefig('model_six_train/Images/fig_regression_Rmsprop_mse.png')
# plt.show()

# print statistical figure of merit
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))


from numpy import linalg as LA
Y_test=np.reshape(Y_test, (Y_test.shape[0],1))
error=y_pred-Y_test

# this is the measure of the prediction performance in percents
error_percentage=LA.norm(error,2)/LA.norm(Y_test,2)*100

plt.figure()
plt.plot(Y_test, 'b', label='Real output')
plt.plot(y_pred,'r', label='Predicted output')
plt.xlabel('Discrete time steps')
plt.ylabel('Output')
plt.legend()
plt.savefig('model_six_train/Images/prediction_offline.png')
plt.show()

# Save model parameters
model_params = dict()
model_params['Xscale'] = X
model_params['Yscale'] = Y
model_params['window'] = window


pickle.dump(model_params, open('model_six_train/model_params.pkl', 'wb'))