#!/usr/bin/env python

#In this course project, you will build a regression model using the deep learning Keras library, and then you will experiment with increasing the number of training epochs and changing number of hidden layers and you will see how changing these parameters impacts the performance of the model.
#Review criteria

#This assignment will be marked by your peers and will be worth 20% of your total grade. The breakdown will be:

#Part A: 5 marks

#Part B: 5 marks

#Part C: 5 marks

#Part D: 5 marks
#Step-By-Step Assignment Instructions

#1. Assignment Topic:

#In this project, you will build a regression model using the Keras library to model the same data about concrete compressive strength that we used in labs 3.

#2. Concrete Data:

#For your convenience, the data can be found here again: https://cocl.us/concrete_data. To recap, the predictors in the data of concrete strength include:

#    Cement

#    Blast Furnace Slag

#    Fly Ash

#    Water

#    Superplasticizer

#    Coarse Aggregate

#    Fine Aggregate

#3. Assignment Instructions:

#Please check the My Submission tab for detailed assignment instructions.

#4. How to submit:

#You will need to submit your code for each part in a Jupyter Notebook. Since each part builds on the previous one, you can submit the same notebook four times for grading. Please make sure that you:

#    use Markdown to clearly label your code for each part,

#    properly comment your code so that your peer who is grading your work is able to understand your code easily,

#    include your comments and discussion of the difference in the mean of the mean squared errors among the different parts.






#A. Build a baseline model (5 marks) 

#Use the Keras library to build a neural network with the following:

#- One hidden layer of 10 nodes, and a ReLU activation function

#- Use the adam optimizer and the mean squared error  as the loss function.

#1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_splithelper function from Scikit-learn.

#2. Train the model on the training data using 50 epochs.

#3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

#4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

#5. Report the mean and the standard deviation of the mean squared errors.

#Submit your Jupyter Notebook with your code and comments.

#B. Normalize the data (5 marks) 

#Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

#How does the mean of the mean squared errors compare to that from Step A?

#C. Increate the number of epochs (5 marks)

#Repeat Part B but use 100 epochs this time for training.

#How does the mean of the mean squared errors compare to that from Step B?

#D. Increase the number of hidden layers (5 marks)

#Repeat part B but use a neural network with the following instead:

#- Three hidden layers, each of 10 nodes and ReLU activation function

#.

#How does the mean of the mean squared errors compare to that from Step B?
#Coursera Honor Code  Learn more










# tutorial on jupiter notebooks:
#
#  https://www.youtube.com/watch?v=h1sAzPojKMg

#Use the Keras library to build a neural network with the following:

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
y = concrete_data['Strength'] # Strength column
n_cols = x.shape[1] # number of predictors

# build the model
model = regression_model()

num_epochs = 50

#4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
mse_list = []
for iter in range(num_epochs):
    #1. Randomly split the data into a training and test sets by holding 30% of the data for testing. 
    #   You can use the train_test_splithelper function from Scikit-learn.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

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
