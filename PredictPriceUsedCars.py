# -*- coding: utf-8 -*-
"""
Created on Sun Mar 1 18:10:16 2020

@author: wchen
"""
# In[1]:
# Importing the libraries
import pandas as pd
import os

# Importing the dataset
os.chdir("/Users/mariamawitanteneh/Desktop/Business Application  of AI")
dataset = pd.read_csv('ToyotaCorolla.csv')

dataset = dataset[['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Automatic', 'Doors', 'Quarterly_Tax',
'Mfr_Guarantee', 'Guarantee_Period', 'Airco', 'Automatic_airco', 'CD_Player',
'Powered_Windows', 'Sport_Model', 'Tow_Bar', 'Price']]
dataset.head()

# In[2]:
# Encoding categorical data
dataset = pd.get_dummies(dataset, columns=['Fuel_Type'], prefix = ['Fuel'])

"""
dataset['Fuel_Type']
Out[84]: 
0       1
1       1
2       1
3       1
4       1
       ..
1431    2
1432    2
1433    2
1434    2
1435    2
Name: Fuel_Type, Length: 1436, dtype: int64
"""
# In[3]:
# Prepare Features and Labels
X = dataset.loc[:,dataset.columns != 'Price']
y = dataset.loc[:,dataset.columns == 'Price']

"""
In [40]: X
Out[40]: 
     Age_08_04     KM Fuel_Type  ... Powered_Windows Sport_Model Tow_Bar
0           23  46986    Diesel  ...               1           0       0
1           23  72937    Diesel  ...               0           0       0
2           24  41711    Diesel  ...               0           0       0
3           26  48000    Diesel  ...               0           0       0
4           30  38500    Diesel  ...               1           0       0
       ...    ...       ...  ...             ...         ...     ...
1431        69  20544    Petrol  ...               1           1       0
1432        72  19000    Petrol  ...               0           1       0
1433        71  17016    Petrol  ...               0           0       0
1434        70  16916    Petrol  ...               0           0       0
1435        76      1    Petrol  ...               0           0       0

[1436 rows x 15 columns]

In [41]: Y
Out[41]: 
0       13500
1       13750
2       13950
3       14950
4       13750
 
1431     7500
1432    10845
1433     8500
1434     7250
1435     6950
Name: Price, Length: 1436, dtype: int64
"""

# In[4]:
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[5]:
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Label Scaling
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

"""
X_train
Out[97]: 
array([[ 0.58557657,  2.81868858, -2.36322149, ..., -1.12441087,
        -0.65818576,  1.57107541],
       [-0.27269589, -0.411118  ,  0.35247247, ...,  0.88935462,
        -0.65818576, -0.63650668],
       [ 0.21008237, -0.18318954,  0.35247247, ..., -1.12441087,
         1.51932793, -0.63650668],
       ...,
       [ 0.74650266,  0.45218529, -2.36322149, ...,  0.88935462,
         1.51932793,  1.57107541],
       [-0.32633792, -0.77558009,  0.35247247, ...,  0.88935462,
        -0.65818576, -0.63650668],
       [ 0.6392186 ,  0.85481723,  0.35247247, ...,  0.88935462,
        -0.65818576, -0.63650668]])
    
X_test
Out[98]: 
array([[ 0.6392186 ,  1.22906782,  0.35247247, ...,  0.88935462,
        -0.65818576,  1.57107541],
       [-0.21905386,  0.81018379,  0.35247247, ...,  0.88935462,
        -0.65818576, -0.63650668],
       [-0.75547414, -1.04332754,  0.35247247, ...,  0.88935462,
         1.51932793, -0.63650668],
       ...,
       [ 0.58557657, -0.71184844,  0.35247247, ...,  0.88935462,
        -0.65818576, -0.63650668],
       [-0.8627582 , -0.96640268,  0.35247247, ..., -1.12441087,
        -0.65818576, -0.63650668],
       [ 0.69286063,  2.29114686, -2.36322149, ..., -1.12441087,
         1.51932793, -0.63650668]])
"""

# In[6]:
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

"""
Epoch 1/20
1148/1148 [==============================] - 0s 235us/step - loss: 0.6734 - accuracy: 0.7091
Epoch 2/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.5000 - accuracy: 0.7117
Epoch 3/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.2752 - accuracy: 0.7117
Epoch 4/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2214 - accuracy: 0.7561
Epoch 5/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2002 - accuracy: 0.9948
Epoch 6/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1848 - accuracy: 0.9956
Epoch 7/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.1715 - accuracy: 0.9956
Epoch 8/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1596 - accuracy: 0.9956
Epoch 9/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1488 - accuracy: 0.9956
Epoch 10/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1388 - accuracy: 0.9956
Epoch 11/20
1148/1148 [==============================] - 0s 111us/step - loss: 0.1297 - accuracy: 0.9956
Epoch 12/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1212 - accuracy: 0.9956
Epoch 13/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1134 - accuracy: 1.0000
Epoch 14/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.1061 - accuracy: 1.0000
Epoch 15/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.0994 - accuracy: 1.0000
Epoch 16/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0932 - accuracy: 1.0000
Epoch 17/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.0874 - accuracy: 1.0000
Epoch 18/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0820 - accuracy: 1.0000
Epoch 19/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.0770 - accuracy: 1.0000
Epoch 20/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.0724 - accuracy: 1.0000
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_20 (Dense)             (None, 5)                 80        
_________________________________________________________________
dense_21 (Dense)             (None, 5)                 30        
_________________________________________________________________
dense_22 (Dense)             (None, 1)                 6         
=================================================================
Total params: 116
Trainable params: 116
Non-trainable params: 0
_________________________________________________________________
"""

# In[7]:
# Part 3 - Making predictions and evaluating the model
from math import sqrt

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.06347389491319165
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.06188155141872638
"""

# In[8]:
# Modify the model to single layer with 5 nodes

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

"""
Epoch 1/20
1148/1148 [==============================] - 0s 235us/step - loss: 0.6734 - accuracy: 0.7091
Epoch 2/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.5000 - accuracy: 0.7117
Epoch 3/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.2752 - accuracy: 0.7117
Epoch 4/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2214 - accuracy: 0.7561
Epoch 5/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2002 - accuracy: 0.9948
Epoch 6/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1848 - accuracy: 0.9956
Epoch 7/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.1715 - accuracy: 0.9956
Epoch 8/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1596 - accuracy: 0.9956
Epoch 9/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1488 - accuracy: 0.9956
Epoch 10/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1388 - accuracy: 0.9956
Epoch 11/20
1148/1148 [==============================] - 0s 111us/step - loss: 0.1297 - accuracy: 0.9956
Epoch 12/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1212 - accuracy: 0.9956
Epoch 13/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1134 - accuracy: 1.0000
Epoch 14/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.1061 - accuracy: 1.0000
Epoch 15/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.0994 - accuracy: 1.0000
Epoch 16/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0932 - accuracy: 1.0000
Epoch 17/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.0874 - accuracy: 1.0000
Epoch 18/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0820 - accuracy: 1.0000
Epoch 19/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.0770 - accuracy: 1.0000
Epoch 20/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.0724 - accuracy: 1.0000
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_20 (Dense)             (None, 5)                 80        
_________________________________________________________________
dense_21 (Dense)             (None, 5)                 30        
_________________________________________________________________
dense_22 (Dense)             (None, 1)                 6         
=================================================================
Total params: 116
Trainable params: 116
Non-trainable params: 0
_________________________________________________________________
"""


# In[9]:
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.05157386947653265
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.04982500227298357
"""

# In[10]:
# Modify the model to two layers with 5 nodes in each layer

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

"""
Epoch 1/20
1148/1148 [==============================] - 0s 235us/step - loss: 0.6734 - accuracy: 0.7091
Epoch 2/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.5000 - accuracy: 0.7117
Epoch 3/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.2752 - accuracy: 0.7117
Epoch 4/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2214 - accuracy: 0.7561
Epoch 5/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.2002 - accuracy: 0.9948
Epoch 6/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1848 - accuracy: 0.9956
Epoch 7/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.1715 - accuracy: 0.9956
Epoch 8/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.1596 - accuracy: 0.9956
Epoch 9/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1488 - accuracy: 0.9956
Epoch 10/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1388 - accuracy: 0.9956
Epoch 11/20
1148/1148 [==============================] - 0s 111us/step - loss: 0.1297 - accuracy: 0.9956
Epoch 12/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1212 - accuracy: 0.9956
Epoch 13/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.1134 - accuracy: 1.0000
Epoch 14/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.1061 - accuracy: 1.0000
Epoch 15/20
1148/1148 [==============================] - 0s 110us/step - loss: 0.0994 - accuracy: 1.0000
Epoch 16/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0932 - accuracy: 1.0000
Epoch 17/20
1148/1148 [==============================] - 0s 108us/step - loss: 0.0874 - accuracy: 1.0000
Epoch 18/20
1148/1148 [==============================] - 0s 107us/step - loss: 0.0820 - accuracy: 1.0000
Epoch 19/20
1148/1148 [==============================] - 0s 109us/step - loss: 0.0770 - accuracy: 1.0000
Epoch 20/20
1148/1148 [==============================] - 0s 106us/step - loss: 0.0724 - accuracy: 1.0000
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_20 (Dense)             (None, 5)                 80        
_________________________________________________________________
dense_21 (Dense)             (None, 5)                 30        
_________________________________________________________________
dense_22 (Dense)             (None, 1)                 6         
=================================================================
Total params: 116
Trainable params: 116
Non-trainable params: 0
_________________________________________________________________
"""


# In[9]:
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.04632696197448908
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.04661587637538337
"""