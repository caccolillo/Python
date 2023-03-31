#!/usr/bin/env python



import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# define regression model
def regression_model():
    # create model with
    # one hidden layer of 10 nodes, and a ReLU activation function
   
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    # Use the adam optimizer and the mean squared error  as the loss function.
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
concrete_data_columns = concrete_data.columns
x = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength

x_norm = (x - x.mean()) / x.std()


y = concrete_data['Strength'] # Strength column
n_cols = x.shape[1] # number of predictors

# build the model
model = regression_model()

num_epochs = 100

#4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
mse_list = []
for iter in range(num_epochs):
    #1. Randomly split the data into a training and test sets by holding 30% of the data for testing. 
    #   You can use the train_test_splithelper function from Scikit-learn.
    x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.3)

    #2. Train the model on the training data using 50 epochs.
    model.fit(x_train, y_train, epochs=num_epochs)

    #3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete 
    #   strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.
    # evaluate the model
    y_pred = model.predict(x_test)   
    mse = mean_squared_error(y_test, y_pred)
    print("Mse = : {} \n ".format(mse))
    mse_list.append(mse)
    
print(mse_list)    

#5. Report the mean and the standard deviation of the mean squared errors.
mean_mse = np.mean(mse_list)
stdv_mse = np.std(mse_list)
print("Mean of the mean squared errors = : {} \n ".format(mean_mse))
print("Standard deviation of the mean squared errors = : {} \n ".format(stdv_mse))
