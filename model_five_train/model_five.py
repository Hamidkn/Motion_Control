import numpy as np
from motion import profile
from plot import draw_plots
import pandas as pd

import keras.models
# import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import keras.api._v1.keras.layers.experimental as ex
from sklearn import preprocessing
from matplotlib import pyplot as plt
import math
import time
from tensorflow.keras.callbacks import LearningRateScheduler
# import tensorflo


initial_learning_rate = 0.0005
epochs = 100
# decay = initial_learning_rate / epochs
def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epochs)

# Create an input pipeline using tf.data
# Next, you will wrap the dataframes with tf.data, in order to shuffle and batch the data
def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("position")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_compiled_model():
    model = Sequential()
    model.add(Dense(40,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['mse'])

    return model


dt=1e-5
t = np.arange(0, 0.5, dt)
u=len(t)
g = 9.81
k = 0.213
step=0.5*1e-3
Sstep= 4*step

x1 = []
x2 = []
curr2 = []
curr1 = []
Ftotal = []
x1 = np.zeros(int(u))
x2 = np.zeros(int(u))
curr1 = np.zeros(int(u))
curr2 = np.zeros(int(u))
Ftotal = np.zeros(int(u))
j = 0

x1, x2, Ftotal, curr1, curr2 = profile(Sstep,dt, u, x1, x2, Ftotal, curr1, curr2)
draw_plots(t,x1,x2,Ftotal,curr1,curr2)

inp = [curr1, curr2, Ftotal, x2, x1]
print(np.shape(inp))

nparray = np.array(inp)
transpose = nparray.transpose()
inp = transpose.tolist()
df = pd.DataFrame(inp, columns=('current1','current2','Ftotal','velocity','position'))

print(df)
print(df.shape)

df.to_csv('model_five_train/dataframe.txt', index=False)

# Split the dataframe into train, validation, and test

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# min_max_scaler = preprocessing.MinMaxScaler()
# train_ds = min_max_scaler.fit_transform(train)
# val_ds = min_max_scaler.fit_transform(val)
# test_ds = min_max_scaler.fit_transform(test)
print(len(train), 'train examples', np.shape(train))
print(len(val), 'validation examples', np.shape(val))
print(len(test), 'test examples', np.shape(test))

# You can see that the dataset returns a dictionary of column names
# (from the dataframe) that map to column values from rows in the dataframe.

batch_size = 32
train_ds = dataframe_to_dataset(train, batch_size=batch_size)
val_ds = dataframe_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = dataframe_to_dataset(test, shuffle=False, batch_size=batch_size)

# print('Every feature:', list(train_features.keys()))
# print('A batch of velocity:', train_features['velocity'])
# print('A batch of targets:', label_batch )


# Numeric columns
# For each of the Numeric feature, you will use a Normalization() layer to
# make sure the mean of each feature is 0 and its standard deviation is 1.
# get_normalization_layer function returns a layer which
# applies featurewise normalization to numerical features.
def get_normalization_layer(name, dataset):
      # Create a Normalization layer for our feature.
  normalizer = ex.preprocessing.Normalization(axis=None)

  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

# photo_count_col = train_features['velocity']
# layer = get_normalization_layer('velocity', train_ds)
# layer(photo_count_col)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['current1','current2','Ftotal','velocity']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)


# Training 


# MAX_EPOCHS = 10

# Create, compile, and train the model
# Now you can create our end-to-end model.
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.Dense(1, activation="tanh")(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer=RMSprop(learning_rate=initial_learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mse"])

# Train the model
start = time.time()
model_info = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=32, 
                callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)])
stop = time.time() - start
print(f'Stop: {stop} s')

# visualize our connectivity graph:
plot_model(model, to_file='model_five_train/Images/model_five.png', show_shapes=True, rankdir="LR")
with open('model_five_train/summarymodel_model_five.txt','w') as file:
            with redirect_stdout(file):
                model.summary()

plt.plot(model_info.history['mse'])
plt.plot(model_info.history['val_mse'])
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train', 'Val'],loc='upper left')
plt.savefig('model_five_train/Images/fig_Accuracy.png')
plt.show()
       
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'],loc='upper left')
plt.savefig('model_five_train/Images/fig_loss.png')
plt.show()

# loss, accuracy = model.evaluate(test_ds)
# print("mse", accuracy)


# nnmodel = Model()
# model = nnmodel.generate_model()
# model.compile(model)
# model.fit(model, inpTrain, outTrain, MAX_EPOCHS)
# model.summary(model)


# model = Sequential()
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(4, activation='relu'))
# model.compile(optimizer='rmsprop',
#                 loss=tf.keras.losses.MeanSquaredError(),
#                 metrics=['mse'])

# history = model.fit(train_data, epochs=MAX_EPOCHS, validation_data=val_data)
# # model.evaluate(inpTest, outTest, verbose = 2)

# with open('model_summary.txt','w') as file:
#            with redirect_stdout(file):
#                 model.summary()

# model.save('model_1.h5')

# model = get_compiled_model()

# model.fit(train_dataset, epochs=MAX_EPOCHS)

# plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True, rankdir="LR")
# with open('summarymodel.txt','w') as file:
#             with redirect_stdout(file):
#                 model.summary()