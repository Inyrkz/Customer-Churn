#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Separating the features and target variable
X = dataset.iloc[:, 3:13].values
print(X)
y = dataset.iloc[:, -1].values
print(y) 

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding the Coutries: France, Spain, Germany, etc
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encoding the Gender: Male and Female
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding the dummy variable trap by removing a country column
X = X[:, 1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing the Keras libraries and packages
import keras
#Initializing the ANN
from keras.models import Sequential
#Initializing the layers in the ANN
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
#output_dim no of hidden units
#init='uniform' keeping initialized weight values close to zero
#using relu activation function for hidden units
#input_dim defining no of input units

classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
#no need for the input_dim because the classifier knows what to expect as input from the previous output layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#using the adam optimizer
#using binary_crossentropy since we have 2 output classes; categorical_crossentropy for 3 classes
#the metric will be a list containing accuracy

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#Predicting the Test set results
y_pred = classifier.predict(X_test)
#setting the threshold to 0.5
y_pred = (y_pred > 0.5)

#making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
