# Stock values prediction and anomalies detection using LSTM neural network

The main porpuse of this project is to predict the future stock values using an LSTM (Long Short Term Memory) neural network. It also detects anomalies (days when the real value of the stock had an worth mentioning deviation from the pedicted value)

## Compilation and Execution


To execute run: ~$python3 forecast_build.py –d \<dataset\> -n \<number of time series selected\>
 

  
And for anomalies detection: ~$python3 detect_build.py –d \<dataset\> -n \<number of time series selected\> -mae \<error\>


## Concepts Used

This project involves the following:

-LSTM neural network

-Numpy, scipy, scikit-learn, matplotlib, tensorflow, pandas, keras, seaborn
