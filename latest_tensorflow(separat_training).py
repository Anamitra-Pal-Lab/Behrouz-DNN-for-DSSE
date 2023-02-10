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
#from keras import regularizers
######################### State estimator######################################
dfx_train = pd.read_csv('temp1.csv')
dfy_train = pd.read_csv('temp2.csv')
x_train = dfx_train.to_numpy()
y_train = dfy_train.to_numpy()
dfx_test = pd.read_csv('temp3.csv')
dfy_test = pd.read_csv('temp4.csv')
x_test = dfx_test.to_numpy()
y_test = dfy_test.to_numpy()

#even_index = np.arange(0,y_test.shape[1],2)
#odd_index = np.arange(1,y_test.shape[1],2) 
#y_train = y_train[:,even_index]
#y_test = y_test[:,even_index]
##### determining PMU location######################################
#pmu_loc_index1 = 4 #index should be based on the table in PPT' don't worry about starting index 0 in python it is taken care of in next lines. use the exact same index from the table in PPT
#pmu_loc_index2 = 15 #index should be based on the table in PPT' don't worry about starting index 0 in python it is taken care of in next lines. use the exact same index from the table in PPT
# pmu_loc_index3 = 15
# #temp1 = np.arange(1,1+6)
# #temp2 = np.arange(x_train.shape[1]/2+1,x_train.shape[1]/2+1+6)
#temp1 = np.arange(pmu_loc_index1*6-5,pmu_loc_index1*6-5 +6)
#temp2 = np.arange(pmu_loc_index1*6-5+x_train.shape[1]/2,pmu_loc_index1*6-5+x_train.shape[1]/2+6)
#temp3 = np.arange(pmu_loc_index2*6-5,pmu_loc_index2*6-5 +6)
#temp4 = np.arange(pmu_loc_index2*6-5+x_train.shape[1]/2,pmu_loc_index2*6-5+x_train.shape[1]/2+6)
# temp5 = np.arange(pmu_loc_index3*6-5,pmu_loc_index3*6-5 +6)
# temp6 = np.arange(pmu_loc_index3*6-5+x_train.shape[1]/2,pmu_loc_index3*6-5+x_train.shape[1]/2+6)
# temp7 = np.arange(1,1+6)
# temp8 = np.arange(x_train.shape[1]/2+1,x_train.shape[1]/2+1+6)
#temp  = np.concatenate((temp3,temp4), axis=0)
# #temp5 = np.concatenate((temp3,temp4), axis=0)
#temp = np.reshape(temp,(1,12))
# #temp5 = np.reshape(temp5,(1,24))
#temp = temp.astype(int)
#temp = temp-1
#temp = temp[0]
#x_train = x_train[:,temp]
#x_test = x_test[:,temp]
#x_train = x_train[:,[0,1,2,3,4,5,12,13,14,15,16,17]]
#x_test = x_test[:,[0,1,2,3,4,5,12,13,14,15,16,17]]
#x_train = x_train[:,[6,7,8,9,10,11,18,19,20,21,22,23]]
#x_test = x_test[:,[6,7,8,9,10,11,18,19,20,21,22,23]]
# x_train = x_train[:,[12,13,14,15,16,17,30,31,32,33,34,35]]
# x_test = x_test[:,[12,13,14,15,16,17,30,31,32,33,34,35]]
#x_train = x_train[:,0:6]
#x_test = x_test[:,0:6]


#targeted_state_y_index = np.arange(1, y_train.shape[1], 2).tolist() #first parameter should be 0 for mag and 1 for angle
targeted_phase_y_index_A = np.array([1,4,7,10,14,17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80])
targeted_phase_y_index_B = np.array([2,5,8,11,13,15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83])
targeted_phase_y_index_C = np.array([3,6,9,12,16,19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82])
#y_train = y_train[:,targeted_state_y_index]
#y_test =y_test [:,targeted_state_y_index]
#y_train = y_train[:,(2*targeted_phase_y_index_C)-1]
#y_test = y_test [:,(2*targeted_phase_y_index_C)-1]









# Build the neural network
# model = Sequential()
#  #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
# model.add(Dense(500, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 2
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 3
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 4
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(500, activation='relu',kernel_initializer='he_normal')) # Hidden 5
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(y_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output

# Build the neural network
# model = Sequential()
#  #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
# model.add(Dense(200, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 2
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 3
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 4
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 5
# model.add(BatchNormalization())
# model.add(Dropout(0.5))


# model.add(Dense(y_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output








#test_acc_list = []
#ASE_list_actual = []
# loss_fn = losses.MeanSquaredError()
# Adam(learning_rate=0.09456, beta_1=0.9, beta_2=0.999, amsgrad=False)
# model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
# #ES = EarlyStopping (monitor='val_loss',patience=20,verbose=1,restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#logdir = "logs/scalars/" #+ datetime.now().strftime("%Y%m%d-%H%M%S")

#logdir = "logs"
#TB = TensorBoard(log_dir=logdir,write_grads=True,histogram_freq=10)
#checkpoint = ModelCheckpoint('weights{epoch:08d}.h5', monitor='loss', verbose=1,save_weights_only=True, mode='auto', period=1)
#history = model.fit(x_train,y_train,verbose=1,epochs=20,validation_split=0.2,callbacks=[ES])

def training (input_train,output_train,input_val,output_val,input_test,output_test,check):
    model = Sequential()
     #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
    model.add(Dense(200, activation='relu', input_dim=input_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 2
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 3
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 4
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 5
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(output_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output
    
    loss_fn = losses.MeanSquaredError()
    #loss_fn = losses.LogCosh()
    Adam(learning_rate=0.09456, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
    #ES = EarlyStopping (monitor='val_loss',patience=20,verbose=1,restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #logdir = "logs/scalars/" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    
    model.fit(input_train,output_train,verbose=1,epochs=1000,validation_data = (input_val,output_val),callbacks=[checkpoint,reduce_lr])
    model.load_weights("weights.best.hdf5")
    pred = model.predict(input_test)
    mag_MAPE_each_node = np.sum(abs((output_test-pred)/output_test),axis = 0)/output_test.shape[0]*100
    phase_mae_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]*180/np.pi
    mag_mae_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]
    
     
    tolerance_interval_input = []
    R2_Score = []
    for i in range(output_test.shape[1]):
        R2_Score.append(r2_score(output_test[:,i], pred[:,i]))
    if check == 'mag':
        temp1 = abs((output_test-pred)/output_test)
        tolerance_interval_input = np.matrix.flatten(temp1)        
        return mag_MAPE_each_node, R2_Score, tolerance_interval_input, mag_mae_each_node
    elif check == 'phase':
        temp1 = abs(pred - output_test)*180/np.pi
        tolerance_interval_input = np.matrix.flatten(temp1)
        return phase_mae_each_node,R2_Score, tolerance_interval_input

All_phases_index = [targeted_phase_y_index_A,targeted_phase_y_index_B,targeted_phase_y_index_C]
All_errors= []
for i in range(len(All_phases_index)):
    y_train_modified = y_train[:,(2*All_phases_index[i])-1] # two -1 for mag and one -1 for ang
    y_test_modified = y_test [:,(2*All_phases_index[i])-1]
    validation_percentage = 20/100
    x_val_modified = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
    y_val_modified = y_train_modified[int(y_train.shape[0]*(1-validation_percentage)):,:] 
    x_train_modified = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
    y_train_modified = y_train_modified[0:int(y_train.shape[0]*(1-validation_percentage)),:] 
    All_errors.append(training(x_train_modified,y_train_modified ,x_val_modified,y_val_modified,x_test,y_test_modified,'phase'))
    print(i)

#history = model.fit(np.reshape(x_train[0:5][:],(5,36)♥),np.reshape(y_train[0:5][:],(5,258)),batch_size=1,verbose=1,epochs=30,validation_split=0.2)

#_, train_acc = model.evaluate(x_train, y_train, verbose=0)
#_, test_acc = model.evaluate(x_test, y_test, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# model.load_weights("weights.best.hdf5")
# start_SE = time()
# pred = model.predict(x_test)
# end_SE = time()
# elapsed_time = end_SE - start_SE
#ASE = np.sum(np.square(pred - y_test))/pred.shape[0]/pred.shape[1]/2 # division by 2 because of number of nodes not the number of states
#test_acc_list.append(ASE)
    
# plot training history
# pyplot.figure()
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='validation')
# pyplot.title('Model loss')
# pyplot.ylabel('Loss')
# pyplot.xlabel('Epoch')
# pyplot.legend(['Train', 'Validation'], loc='upper left')


# pyplot.figure()
# pyplot.plot(history.history['mae'])
# pyplot.plot(history.history['val_mae'])
# pyplot.title('Model MAE')
# pyplot.ylabel('MAE')
# pyplot.xlabel('Epoch')
# pyplot.legend(['Train', 'Test'], loc='upper left')



#pyplot.show()


#pred = model.predict(x_test)
#score = np.sqrt(metrics.mean_squared_error(pred,y_test))
#print(f"Final score (RMSE): {score}")
#print("Shape: {}".format(pred.shape))
#print(pred)

# mag_pred = np.array(list(range(pred.shape[0])))
# phase_pred = np.array(list(range(pred.shape[0])))
# mag_y_test = np.array(list(range(pred.shape[0])))
# phase_y_test = np.array(list(range(pred.shape[0])))

# mag_pred = np.reshape(mag_pred, (pred.shape[0], 1))
# phase_pred = np.reshape(phase_pred, (pred.shape[0], 1))
# mag_y_test = np.reshape(mag_y_test, (pred.shape[0], 1))
# phase_y_test = np.reshape(phase_y_test, (pred.shape[0], 1))

# for i in range(pred.shape[1]):
#     if i%2 ==0:
#         mag_pred = np.concatenate((mag_pred, pred[:,[i]]),1)
#         mag_y_test = np.concatenate((mag_y_test, y_test[:,[i]]),1)
#     else     :  
#         phase_pred = np.concatenate((phase_pred, pred[:,[i]]),1)
#         phase_y_test = np.concatenate((phase_y_test, y_test[:,[i]]),1)

# mag_pred = mag_pred [:,1:mag_pred.shape[1]]
# phase_pred = phase_pred[:,1:phase_pred.shape[1]]
# mag_y_test = mag_y_test [:,1:mag_y_test.shape[1]]  
# phase_y_test = phase_y_test  [:,1:phase_y_test.shape[1]] 

# even_index = np.arange(0,y_test.shape[1],2)
# odd_index = np.arange(1,y_test.shape[1],2)     
# phase_MAE = mean_absolute_error(pred[:,odd_index], y_test[:,odd_index])*180/np.pi  
# mag_MAPE= np.sum(abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index]))/y_test[:,even_index].shape[0]/(y_test[:,even_index].shape[1])*100

# temp1 = abs(pred[:,odd_index]- y_test[:,odd_index])*180/np.pi
# temp2 = abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index])
# tolerance_interval_input_angle = np.matrix.flatten(temp1)
# tolerance_interval_input_mag   = np.matrix.flatten(temp2)

# # when trainig for all together
# mag_MAPE_each_node = np.sum(abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index]),axis = 0)/y_test[:,even_index].shape[0]*100
# phase_mae_each_node = np.sum(abs(y_test[:,odd_index]-pred[:,odd_index]),axis = 0)/y_test[:,odd_index].shape[0]*180/np.pi
# # when training for a particular phase and parameter
# mag_MAPE_each_node = np.sum(abs((y_test-pred)/y_test),axis = 0)/y_test.shape[0]*100
# phase_mae_each_node = np.sum(abs(y_test-pred),axis = 0)/y_test.shape[0]*180/np.pi

# pyplot.figure()
# pyplot.plot(np.arange(0, mag_MAPE_each_node.shape[0], 1), mag_MAPE_each_node,'bo')
# pyplot.ylabel('magnitude MAPE')
# pyplot.xlabel('node number')
# pyplot.figure()
# pyplot.plot(np.arange(0, phase_mae_each_node.shape[0], 1), phase_mae_each_node,'ro')
# pyplot.ylabel('phase MAE')
# pyplot.xlabel('node number')


#phase_A_angle_MAE_for_each_node = phase_mae_each_node[targeted_phase_y_index_A-1]
#phase_A_mag_MAPE_for_each_node = mag_MAPE_each_node[targeted_phase_y_index_A-1]

# R2_Score = []
# for i in odd_index:
#     R2_Score.append(r2_score(y_test[:,i], pred[:,i]))


# R2_Score = []
# for i in range(y_test.shape[1]):
#     R2_Score.append(r2_score(y_test[:,i], pred[:,i]))


######### bad data detection #############
# dfx_test_bad = pd.read_csv('temp5.csv')
# x_test_bad = dfx_test_bad.to_numpy()

# false_alarm_level_1 = 2.5758  #  1% level percentage
# false_alarm_level_2 = 2.3263  #  2% level percentage
# false_alarm_level_3 = 2.1701  #  3% level percentage
# false_alarm_level_4 = 2.0537  #  4% level percentage
# false_alarm_level_5 = 1.9600  #  5% level percentage
# bad_data_index = np.where(abs(x_test_bad)  > false_alarm_level_5)
# x_test_bad_replaced = np.copy(x_test_bad)
# x_test_bad_replaced[bad_data_index[0],bad_data_index[1]] = 0
# pred_bad = model.predict(x_test_bad_replaced)
# phase_MAE_bad = mean_absolute_error(pred_bad[:,odd_index], y_test[:,odd_index])*180/np.pi  
# mag_MAPE_bad = np.sum(abs((y_test[:,even_index]-pred_bad[:,even_index])/y_test[:,even_index]))/y_test[:,even_index].shape[0]/(y_test[:,even_index].shape[1])*100

# mag_MAPE_each_node_bad = np.sum(abs((y_test[:,even_index]-pred_bad[:,even_index])/y_test[:,even_index]),axis = 0)/y_test[:,even_index].shape[0]*100
# phase_mae_each_node_bad = np.sum(abs(y_test[:,odd_index]-pred_bad[:,odd_index]),axis = 0)/y_test[:,odd_index].shape[0]*180/np.pi
# pyplot.figure()
# pyplot.plot(np.arange(0, mag_MAPE_each_node_bad.shape[0], 1), mag_MAPE_each_node_bad,'bo')
# pyplot.ylabel('magnitude MAPE bad')
# pyplot.xlabel('node number')
# pyplot.figure()
# pyplot.plot(np.arange(0, phase_mae_each_node_bad.shape[0], 1), phase_mae_each_node_bad,'ro')
# pyplot.ylabel('phase MAE bad')
# pyplot.xlabel('node number')

######### bad data detection #############

######### missing data handling #############
#### 6 Scenarios from power point, slide 45####
# x_test_missing = np.copy(x_test)
# n_scen = 6
# #portion_per_scen = np.floor(2496/n_scen) 
# portion_per_scen = np.split(np.arange(0,2496),n_scen) # 2496 is close to 2500 which is the x_test size. the reasons is that it has to be devidedable by 6
# #portion_per_scen = np.split(np.arange(0,750),n_scen) # 750 is 30% of the whole test size
# # 1. first mPMU is completely missing
# closest_norm_index = []
# temp_index = np.array([6,7,8,9,10,11,18,19,20,21,22,23])
# for i in portion_per_scen[0]:
#     temp = np.argmin(np.linalg.norm(x_test[i,temp_index]-x_train[:,temp_index],axis=1))   
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[0][:,None],np.array([0,1,2,3,4,5,12,13,14,15,16,17])] = x_train[closest_norm_index[:,None],np.array([0,1,2,3,4,5,12,13,14,15,16,17])] # replacement with L2 norm
# #x_test_missing[portion_per_scen[0][:,None],np.array([0,1,2,3,4,5,12,13,14,15,16,17])] =  np.zeros((int(750/n_scen),12)) # replacement with expected value

# # 2. secondm mPMU is completely missing
# closest_norm_index = []
# temp_index = np.array([0,1,2,3,4,5,12,13,14,15,16,17])
# for i in portion_per_scen[1]:
#     temp = np.argmin(np.linalg.norm(x_test[i,temp_index]-x_train[:,temp_index],axis=1))    
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[1][:,None],np.array([6,7,8,9,10,11,18,19,20,21,22,23])] = x_train[closest_norm_index[:,None],np.array([6,7,8,9,10,11,18,19,20,21,22,23])]  # replacement with L2 norm
# #x_test_missing[portion_per_scen[1][:,None],np.array([6,7,8,9,10,11,18,19,20,21,22,23])] = np.zeros((int(750/n_scen),12)) # replacement with expected value


# #3. first mPMU voltage is completely missing
# closest_norm_index = []
# for i in portion_per_scen[2]:
#     temp = np.argmin(np.linalg.norm(x_test[i,6:]-x_train[:,6:],axis=1))   
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[2],0:6] = x_train[closest_norm_index,0:6]
# #x_test_missing[portion_per_scen[2],0:6] = np.zeros((int(750/n_scen),6)) 

# #4. first mPMU current is completely missing
# closest_norm_index = []
# temp_index = np.array([0,1,2,3,4,5,6,7,8,9,10,11,18,19,20,21,22,23])
# for i in portion_per_scen[3]:
#     temp = np.argmin(np.linalg.norm(x_test[i,temp_index]-x_train[:,temp_index],axis=1))   
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[3],12:18] = x_train[closest_norm_index,12:18]
# #x_test_missing[portion_per_scen[3],12:18] = np.zeros((int(750/n_scen),6)) 


# #5. second mPMU voltage is completely missing
# closest_norm_index = []
# temp_index = np.array([0,1,2,3,4,5,12,13,14,15,16,17,18,19,20,21,22,23])
# for i in portion_per_scen[4]:
#     temp = np.argmin(np.linalg.norm(x_test[i,temp_index]-x_train[:,temp_index],axis=1))   
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[4],6:12] = x_train[closest_norm_index,6:12]
# #x_test_missing[portion_per_scen[4],6:12] = np.zeros((int(750/n_scen),6)) 

# #6. second mPMU current is completely missing
# closest_norm_index = []
# for i in portion_per_scen[5]:
#     temp = np.argmin(np.linalg.norm(x_test[i,0:18]-x_train[:,0:18],axis=1))   
#     closest_norm_index.append(temp)
# closest_norm_index = np.array(closest_norm_index)    
# x_test_missing[portion_per_scen[5],18:] = x_train[closest_norm_index,18:]
# #x_test_missing[portion_per_scen[5],18:] = np.zeros((int(750/n_scen),6))



# pred_missing = model.predict(x_test_missing)
# phase_MAE_missing_data = mean_absolute_error(pred_missing[:,odd_index], y_test[:,odd_index])*180/np.pi  
# mag_MAPE_missing_data= np.sum(abs((y_test[:,even_index]-pred_missing[:,even_index])/y_test[:,even_index]))/y_test[:,even_index].shape[0]/(y_test[:,even_index].shape[1])*100
# phase_MAE_missing_data_each_sample = abs(y_test[:,odd_index]-pred_missing[:,odd_index])*180/np.pi
# phase_MAE_each_sample = abs(y_test[:,odd_index]-pred[:,odd_index])*180/np.pi

# phase_MAE_only_missing_data = mean_absolute_error(pred_missing[0:750,odd_index], y_test[0:750,odd_index])*180/np.pi  
# mag_MAPE_only_missing_data= np.sum(abs((y_test[0:750,even_index]-pred_missing[0:750,even_index])/y_test[0:750,even_index]))/y_test[0:750,even_index].shape[0]/(y_test[0:750,even_index].shape[1])*100



# pyplot.figure()
# pyplot.plot(phase_MAE_missing_data_each_sample[:,83])
#############################################



####### calculating error for islanded topology comment for first training then run it line be line manually
# dfx_test = pd.read_csv('temp3.csv')
# dfy_test = pd.read_csv('temp4.csv')
# x_test = dfx_test.to_numpy()
# y_test = dfy_test.to_numpy()

# pred = model.predict(x_test)

# temp1 = []
# temp2 = []
# temp1 = np.mean(y_test,0)
# temp2 = np.where(temp1 ==0)
# y_test = np.delete(y_test,temp2,1)
# pred = np.delete(pred,temp2,1)

# odd_index = np.arange(1,y_test.shape[1],2)
# even_index = np.arange(0,y_test.shape[1],2)     
# phase_MAE = mean_absolute_error(pred[:,odd_index], y_test[:,odd_index])*180/np.pi  
# mag_MAPE= np.sum(abs((y_test[:,even_index]-pred[:,even_index])/y_test[:,even_index]))/y_test[:,even_index].shape[0]/(y_test[:,even_index].shape[1])*100
#####################################################

# phase_MAE_normalized_pi = mean_absolute_error(phase_y_test, phase_pred)  
# phase_MAE_normalized_degree = phase_MAE_normalized_pi*180/3.1415926535
# mag_MAPE_normalized = np.sum(abs((mag_y_test-mag_pred)/mag_y_test))/mag_y_test.shape[0]/(mag_y_test.shape[1])*100
# mag_MAE_normalized = mean_absolute_error(mag_y_test, mag_pred)   
    
#    df_mu_sigma = pd.read_csv('temp5.csv')    
#    mu_sigma = np.transpose(df_mu_sigma.to_numpy())
#    df_test_All_Polar_Voltages = pd.read_csv('temp6.csv')
#    test_All_Polar_Voltages = df_test_All_Polar_Voltages.to_numpy()
#    actual_value = np.array(list(range(pred.shape[0])))
#    actual_value = np.reshape(actual_value, (pred.shape[0], 1))
#    for i in range(pred.shape[1]):
#        actual_value = np.concatenate((actual_value,pred[:,[i]]*mu_sigma[1][i] + mu_sigma[0][i]),1)
#    actual_value = np.delete(actual_value,0,1)
#    
#    odd = []
#    even = []
#    for i in range(pred.shape[1]):
#        if i%2 ==0:
#           even.append(i)
#        else:
#            odd.append(i)
#    mag_MAE = mean_absolute_error(actual_value[:, even],test_All_Polar_Voltages [:, even])
#    mag_MAPE_normalized = np.sum(abs((mag_y_test-mag_pred)/mag_y_test))/mag_y_test.shape[0]/(mag_y_test.shape[1])*100
#    phase_MAE = mean_absolute_error(actual_value[:,odd],test_All_Polar_Voltages [:, odd])
#    ASE_actual = np.sum(np.square(actual_value - test_All_Polar_Voltages))/pred.shape[0]/pred.shape[1]/2  # division by 2 because of number of nodes not the number of states
#    ASE_list_actual.append(ASE_actual)

#     calculating error for some buses refer to the excel file
#    MAE_bus_index_mag   = [24,96,264]
#    MAE_bus_index_phase = [25,97,265]
#    MAE_each_node_mag = []
#    MAE_each_node_phase = []
#    for i in MAE_bus_index_mag:
#        MAE_each_node_mag.append(mean_absolute_error(actual_value[:, i],test_All_Polar_Voltages [:, i]))
#    for i in MAE_bus_index_phase:
#         MAE_each_node_phase.append(mean_absolute_error(actual_value[:, i],test_All_Polar_Voltages [:, i]))
    
    
#weights_model = model.get_weights()    
#replaydata = ReplayData(x_train, y_train, filename='name1.h5', group_name='group1', model=model)
#f = h5py.File('model_weights.hdf5', 'r')

# def step(real_x, real_y):
#     with tf.GradientTape() as tape:
#         # Make prediction
#         pred_y = model(real_x.reshape((-1, 28, 28, 1)))
#         # Calculate loss
#         model_loss = tf.keras.losses.categorical_crossentropy(real_y, pred_y)
    
#     # Calculate gradients
#     model_gradients = tape.gradient(model_loss, model.trainable_variables)
#     # Update model
#     optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))





