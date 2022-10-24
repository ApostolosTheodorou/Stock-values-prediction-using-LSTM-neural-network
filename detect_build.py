# Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
from tensorflow.keras.models import load_model
import seaborn as sns
import sys

# Define the lookback parameter
lookback= int(60)

# Read and initialize the command line arguments
for i, arg in enumerate(sys.argv):
    if sys.argv[i] == "-d":
        dataset= sys.argv[i+1]
    elif sys.argv[i] == "-n":
        numberOfTimeseries= int(sys.argv[i+1])
    elif sys.argv[i] == "-mae":
        mae= float(sys.argv[i+1])

# Read the csv file
dataframe=pd.read_csv(dataset, sep='\t', header=None)
dataframe= dataframe.transpose()
dataframe_size= len(dataframe)



# Define the size of the training set as 80% of total dataset size
last_train_day= int(0.8 * dataframe_size)


# Create a list of the days
days=[]
for i in range(0, dataframe_size):
    days.append(i)

# Insert the days in the dataframe as the first column
dataframe['Days']= days
first_column= dataframe.pop('Days')
dataframe.insert(0, 'Days', first_column)



# Input data from all time series (currently empty)
general_X_train= np.empty((last_train_day-lookback, lookback, 1))
# Labels for the training (currently empty)
general_y_train= np.empty((last_train_day-lookback))

# For every time serie
for i in range (1, numberOfTimeseries):
    # Take a dataframe with the columns of 1)days and 2)current time serie
    df = dataframe[['Days', i-1]]

    # Separate the current time serie in training set(80%) and test set(20%)
    training_set= df.iloc[1:last_train_day, 1].values
    test_set= df.iloc[last_train_day:, 1].values
    

    # Normalize the input using Standard Scaler
    scaler = StandardScaler()
    training_set= np.reshape(training_set, (-1,1))
    test_set= np.reshape(test_set, (-1,1))
    #print("training set shape 2222", training_set.shape)
    scaler = scaler.fit(training_set)
    training_set_scaled = scaler.transform(training_set)
    test_set_scaled = scaler.transform(test_set)
    
    # Create the input sequences and the labels for the training of the model
    X_train = []
    y_train = []
    for j in range(lookback, last_train_day-1):
        X_train.append(training_set_scaled[j-lookback:j, 0])
        y_train.append(training_set_scaled[j, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # (2860, 60, 1)
    general_X_train= np.concatenate((general_X_train, X_train), axis=0)
    general_y_train= np.concatenate((general_y_train, y_train), axis=0)

# Define the model to be trained
model_B = Sequential()
model_B.add(LSTM(units=32, input_shape=(X_train.shape[1], 1)))
model_B.add(Dropout(rate=0.25))
model_B.add(RepeatVector(X_train.shape[1]))
model_B.add(LSTM(units=32, return_sequences=True))
model_B.add(Dropout(rate=0.25))
model_B.add(TimeDistributed(Dense(1)))

# Specify the optimizer and the learning rate
opt = Adam(learning_rate=0.0001)
# Compile the model
model_B.compile(optimizer = opt, loss = 'mae')

# fit model
history = model_B.fit(general_X_train, general_y_train, epochs=5, batch_size=8, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Save the model 
model_B.save('models/model_B_ep5_bs8_lb60_in100_ler000001_ls32-R-32_do025.h5')

for i in range(1, numberOfTimeseries):
    # Getting the predicted stock price
    dataset_train = dataframe.iloc[1:last_train_day, i:i+1]
    dataset_test = dataframe.iloc[last_train_day:, i:i+1]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - lookback:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
        
    X_test = []
    for j in range(lookback, lookback+len(dataset_test)):
        X_test.append(inputs[j-lookback:j, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    #print(X_test.shape)
    # (730, 60, 1)

    predicted_stock_price = model_B.predict(X_test)
   

    testMAE = np.mean(np.abs(predicted_stock_price - X_test), axis=1)
    plt.hist(testMAE, bins=30)
    plt.show()

    mae= float(0.85 * (np.amax(testMAE, axis=None) - np.amin(testMAE, axis=None)) + np.amin(testMAE, axis=None))

    
    anomalies = pd.DataFrame(dataset_test)
    anomalies['testMAE'] = testMAE
    anomalies['mae'] = mae
    anomalies['anomaly'] = anomalies['testMAE'] > anomalies['mae']
    
    
    #Plot testMAE vs max_trainMAE
    sns.lineplot(x=dataframe.iloc[last_train_day:,0], y=anomalies['testMAE'])
    sns.lineplot(x=dataframe.iloc[last_train_day:,0], y=anomalies['mae'])
    plt.show()

    anomalies_set = anomalies.loc[anomalies['anomaly'] == True]

    #Plot anomalies
    sns.lineplot(x=dataframe.iloc[last_train_day:,0], y=anomalies.iloc[:,0])
    sns.scatterplot(x=dataframe.iloc[last_train_day:,0], y=anomalies_set.iloc[:,0], color='r')
    plt.show()
    
