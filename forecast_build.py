# Import the necessary libraries
import math
from time import time
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys


# Command line arguments
dataset= "NONE"  # The dataset of stocks that will be given by the user as input 
numberOfTimeseries= int(0) # The subset of stocks that will be used to train the model and predict future values

# Read and initialize the command line arguments
for i, arg in enumerate(sys.argv):
    if sys.argv[i] == "-d":
        dataset= sys.argv[i+1]
    elif sys.argv[i] == "-n":
        numberOfTimeseries= int(sys.argv[i+1])

# Read the csv file
df=pd.read_csv(dataset, sep='\t')

#Bring the data in the correct form
df= df.head(numberOfTimeseries)
df= df.transpose()

# Split the dataset into training and test set
# Training set consists of the 80% of the subset of the time-series
# Test set consists of 20% of the subset of the time-series
last_train_day= (8 * df.shape[0]) // 10


general_X_train= np.empty((2860, 60, 1))
general_y_train= np.empty((2860))
# Train the model for every timeserie separately
for i in range(0, numberOfTimeseries):
    training_set= df.iloc[1:last_train_day, i:i+1].values
    test_set= df.iloc[last_train_day:, i:i+1].values

    # Normalize the input
    # Scale down to range [0-1]
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    # Create 2860 (80% * values - 60) arrays with 60 values as a training input - X_train
    # Create 2860 correct future values to train the model - y_train
    X_train = []
    y_train = []
    for j in range(60, last_train_day-1):
        X_train.append(training_set_scaled[j-60:j, 0])
        y_train.append(training_set_scaled[j, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # (2860, 60, 1)
    general_X_train= np.concatenate((general_X_train, X_train), axis=0)
    general_y_train= np.concatenate((general_y_train, y_train), axis=0)

# Define the model to be trained
model_A2 = Sequential()#Adding the first LSTM layer and some Dropout regularisation
model_A2.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model_A2.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
model_A2.add(LSTM(units = 64, return_sequences = True))
model_A2.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
model_A2.add(LSTM(units = 64, return_sequences = True))
model_A2.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
model_A2.add(LSTM(units = 64))
model_A2.add(Dropout(0.2))# Adding the output layer
model_A2.add(Dense(units = 1))

# Specify the optimizer and the learning rate
opt = Adam(learning_rate=0.0001)
# Compile the model
model_A2.compile(optimizer = opt, loss = 'mean_squared_error')
# Fitting the model to the training set
model_A2.fit(general_X_train, general_y_train, epochs = 5, batch_size = 32)

# Save the model 
model_A2.save('models/model_A2_ep5_bs32_lb60_in100_ler0001.h5')

for i in range(0, numberOfTimeseries):
    # Getting the predicted stock price
    dataset_train = df.iloc[1:last_train_day, i:i+1]
    dataset_test = df.iloc[last_train_day:, i:i+1]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
        
    X_test = []
    for j in range(60, 790):
        X_test.append(inputs[j-60:j, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  
    # (730, 60, 1)

    predicted_stock_price = model_A2.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    time_axis=[]
    for j in range(1, 732):
        time_axis.append(j)
    
    time_axis= np.array(time_axis)
    
    

    # Visualising the results
    company= df.iloc[0,i]
    plt.plot(time_axis[: ], dataset_test.values, color = 'red', label = 'Real Stock Price')
    plt.plot(time_axis[:730 ], predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.xticks(np.arange(0,730,100))
    plt.title(company)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
