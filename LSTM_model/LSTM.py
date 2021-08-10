import csv
import keras.models
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tqdm.keras import TqdmCallback
import math, sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.feature_selection import SelectKBest, f_regression


initial_learning_rate = 0.01
epochs = 150
decay = initial_learning_rate / epochs
window = 15
layers = 2
units = 100
drop = 0.1

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)


dataframe = 'LSTM_model/dataframe1/dataframe_main_model.txt'

df = pd.read_csv(dataframe)
print(df)
# dataset = df.values
X = df[['Ftotal','velocity']].values
Y = df[['position']].values


# min_max_scaler = preprocessing.MinMaxScaler()
# X_scale = min_max_scaler.fit_transform(dataset)
s_x = MinMaxScaler()
X = s_x.fit_transform(X)
s_y = MinMaxScaler()
Y = s_y.fit_transform(Y)

# X = X_scale[:,2:4]
# Y = X_scale[:,4]

# Each input uses last 'window' number of Tsp and err to predict the
X_lstm = []
Y_lstm = []
for i in range(window,len(df)):
    X_lstm.append(X[i-window:i])
    Y_lstm.append(Y[i])

X_lstm, Y_lstm = np.array(X_lstm), np.array(Y_lstm)
# min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_lstm, Y_lstm, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# Model
model = Sequential()
if layers == 1:
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(drop))

else:
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(drop))

    for i in range(layers - 2):
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(drop))
    
    model.add(LSTM(units))
    model.add(Dropout(drop))

model.add(Dense(1))

model.compile(loss=tf.keras.losses.MeanSquaredError(), 
        optimizer =RMSprop(learning_rate=initial_learning_rate), 
        metrics=['mse'])

es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)

start = time.time()
# model_info = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_data=(X_val, Y_val))
model_info = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_data=(X_val, Y_val),
                callbacks=[es,TqdmCallback(verbose=1)])
stop = time.time()
print(f'Training time: {stop - start} s')
plot_model(model, to_file='LSTM_model/Images/LSTM_model_RMSPROP.png', show_shapes=True, rankdir="LR")
with open('LSTM_model/model/summarymodel_LSTM_model.txt','w') as file:
            with redirect_stdout(file):
                model.summary()
model.save('LSTM_model/model/LSTM_model.h5')

# show results and save
epochs = es.stopped_epoch
plt.plot(model_info.history['loss'], label='loss')
plt.plot(model_info.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LSTM_model/Images/LSTM_model_fig_Accuracy_Rmsprop_loss.png')
plt.show()



loss, mse = model.evaluate(X_test, Y_test)
print(f'mse: {mse}, loss: {loss}')

y_pred = model.predict(X_test)
# Unscale data
ytest_us = s_y.inverse_transform(Y_test)
# Xtest_us = min_max_scaler.inverse_transform(X_test[:,-1,:])
# ytest_us = min_max_scaler.inverse_transform(Y_test)
# yp = min_max_scaler.inverse_transform(y_pred)
# sp = Xtest_us[:,1]

# plt.plot(sp,'r-',label='$T_1$ $(^oC)$')

regressor = LinearRegression()
regressor.fit(Y_test.reshape(-1, 1), y_pred)
y_fit = regressor.predict(y_pred)

plt.plot(model_info.history['mse'])
plt.plot(model_info.history['val_mse'])
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train','Val'],loc='upper right')
plt.savefig('LSTM_model/Images/LSTM_model_fig_Accuracy_Rmsprop_mse.png')
plt.show()
       

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(Y_test, y_pred, color='blue', label= 'data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.savefig('LSTM_model/Images/LSTM_model_fig_regression_Rmsprop_mse.png')
plt.show()

# print statistical figure of merit
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))


from numpy import linalg as LA
Y_test=np.reshape(Y_test, (Y_test.shape[0],1))
error=y_pred-Y_test

# this is the measure of the prediction performance in percents
error_percentage=LA.norm(error,2)/LA.norm(Y_test,2)*100

plt.plot(Y_test,'b-',label='${position}, real output$')
plt.plot(y_pred,'g-',label='${LSTM}$')
plt.legend(fontsize=12,loc='lower right')
plt.xlabel('Time',size=14)
plt.ylabel('Value',size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('LSTM_model/Images/LSTM_model_fig_testLSTM.png')
plt.show()

# Save model parameters
model_params = dict()
model_params['Xscale'] = X
model_params['Yscale'] = Y
model_params['window'] = window


pickle.dump(model_params, open('LSTM_model/model/LSTM_model_params.pkl', 'wb'))