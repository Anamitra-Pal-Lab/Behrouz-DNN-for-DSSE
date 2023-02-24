from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from time import time 
from datetime import datetime
import os
#from deepreplay.replay import Replay
#from deepreplay.callbacks import ReplayData
from tensorflow.keras.backend import gradients
import tensorflow as tf
#from keras.callbacks import ModelCheckpoint
import h5py
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import r2_score
import random as python_random
from tensorflow.keras import losses
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

######################### State estimator######################################
dfx_train = pd.read_csv('temp1.csv')
dfy_train = pd.read_csv('temp2.csv')
x_train = dfx_train.to_numpy()
y_train = dfy_train.to_numpy()
dfx_test = pd.read_csv('temp3.csv')
dfy_test = pd.read_csv('temp4.csv')
x_test = dfx_test.to_numpy()
y_test = dfy_test.to_numpy()



#targeted_state_y_index = np.arange(1, y_train.shape[1], 2).tolist() #first parameter should be 0 for mag and 1 for angle
targeted_phase_y_index_A = np.array([1,4,7,10,14,17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80])
targeted_phase_y_index_B = np.array([2,5,8,11,13,15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83])
targeted_phase_y_index_C = np.array([3,6,9,12,16,19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82])
#y_train = y_train[:,targeted_state_y_index]
#y_test =y_test [:,targeted_state_y_index]
### choose phase and state type ####
# phases: A, B, C
# states: magnitude, angle

# for magnitude estimation (comment angle estimation)
y_train = y_train[:,(2*targeted_phase_y_index_A-1)-1] 
y_test = y_test [:,(2*targeted_phase_y_index_A-1)-1]

# for angle estimation (comment magnitude estimation)
# y_train = y_train[:,(2*targeted_phase_y_index_A)-1]  
# y_test = y_test [:,(2*targeted_phase_y_index_A)-1]



validation_percentage = 20/100
x_val = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
y_val = y_train[int(y_train.shape[0]*(1-validation_percentage)):,:] 
x_train = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
y_train = y_train[0:int(y_train.shape[0]*(1-validation_percentage)),:]





# Build the neural network
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 2
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 3
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 4
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 5
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(y_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output





loss_fn = losses.MeanSquaredError()
Adam(learning_rate=0.09456, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(x_train,y_train,verbose=1,epochs=2000,validation_data = (x_val,y_val),callbacks=[checkpoint,reduce_lr])
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
model.load_weights("weights.best.hdf5")
start_SE = time()
pred = model.predict(x_test)
end_SE = time()
elapsed_time = end_SE - start_SE

# plot training history
pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Validation'], loc='upper left')



even_index = np.arange(0,y_test.shape[1],2)
odd_index = np.arange(1,y_test.shape[1],2)     
phase_MAE = mean_absolute_error(pred[:,odd_index], y_test[:,odd_index])*180/np.pi  
phase_MAE = mean_absolute_error(pred, y_test)*180/np.pi  
mag_MAPE= np.sum(abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index]))/y_test[:,even_index].shape[0]/(y_test[:,even_index].shape[1])*100
mag_MAPE= np.sum(abs((y_test-pred)/y_test))/y_test.shape[0]/(y_test.shape[1])*100

temp1 = abs(pred[:,odd_index]- y_test[:,odd_index])*180/np.pi
temp2 = abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index])


mag_MAPE_each_node = np.sum(abs((y_test-pred)/y_test),axis = 0)/y_test.shape[0]*100
# phase_mae_each_node = np.sum(abs(y_test-pred),axis = 0)/y_test.shape[0]*180/np.pi

pyplot.figure()
pyplot.plot(np.arange(0, mag_MAPE_each_node.shape[0], 1), mag_MAPE_each_node,'bo')
pyplot.ylabel('magnitude MAPE')
pyplot.xlabel('node number')
pyplot.figure()
# pyplot.plot(np.arange(0, phase_mae_each_node.shape[0], 1), phase_mae_each_node,'ro')
# pyplot.ylabel('phase MAE')
# pyplot.xlabel('node number')



R2_Score = []
for i in range(y_test.shape[1]):
    R2_Score.append(r2_score(y_test[:,i], pred[:,i]))






print('min_val_loss: ',min(history.history['val_loss']))
print('min_val_MAE: ',min(history.history['val_MAE']))







