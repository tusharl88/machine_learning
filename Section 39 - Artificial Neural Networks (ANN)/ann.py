# Artificial Neural Network
#wheneve you open this program and find any confusion always restart kernel by going to settings present in top right of console.
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# hit ctrl + enter.

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#hit ctrl+enter

#now separating depndent and independent variable.
X = dataset.iloc[:, 3:13].values   #here have used 3 to 13 because 3 is included but index 13 is excluded, therefore index 12 will be included
y = dataset.iloc[:, 13].values
#hit ctrl + enter
#then dataset will be displayed
# go to variable explorer and click on dataset. last column is of exited.
#our goal is to find the last column of dependent variable by using other columns of 
#independent variable.
#in variable explorer we can clealry see that dataset X is of object type.
#This is because we have categorical features in our X dataset which we need to encode into 0 and 1.


# Encoding categorical data
#since we have two encode two independent variable which are geography and gender.
#so we require two objects here labelencoder_X_1 and labelencoder_X_2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  #colon is for rows and 1 is for geography as we can see in out conseole that geography(like france) are at index 1.
#hit ctrl +enter


labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #similarily 2 for gender.
#hit ctrl +enter


#creating dummy variable
#here we are creating dummy variable for only column geography and not for gender because it has only two features of male and female.
# while geography has three features of spain, france, germany.
onehotencoder = OneHotEncoder(categorical_features = [1])   #1 is the index for geography for which we are producing dummy variable.
X = onehotencoder.fit_transform(X).toarray()
#hit ctrl + enter.
#clearly object type has changed.

#removing dummy variable trap.
#now in order to remove dummy variable trap we will remove the 1st column of geography(index 0) and keep the 2nd column of geography with index 2. 
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split   #here create_validation is removed by model selection otherwise we get warning in console.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#hit ctrl+enter

# Feature Scaling
#feature scaling is applied in deep learning because it involve huge calculations.(it was not used in linear regression)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#this is the end of data_preprocessing step.



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras          #we have already installed keras in the system and now we are importing it in spyder.
#hit ctrl+enter and you will see tensorflow backend inside console

from keras.models import Sequential #sequential model helps to initialize neural network.
#hit ctrl + enter

from keras.layers import Dense  #it build layers of our neural network.
#hit ctrl +enter.

# Initialising the ANN
#There are two ways of initialising or defining ANN 
#1.by defining sequence of layers or
#2.by defining graph
#we will use 1.
classifier = Sequential()   #here we are creating an object classifier.
#using sequential class by using sequential module. This object 'classifier' is itself a model or future neural network whose role will be of classifier
#hit ctrl +enter.




# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim= 6, init = 'uniform', activation = 'relu', input_dim = 11))
#output_dim means number of nodes we want to add to hidden layer(not input layer).

#number of nodes to be added to the hidden layer is calculated by two ways:
#1. finding the average of number of nodes in input layer  and the output layer.i.e. 11+1/2 =6
#2. by using parameter chinning method. Here a separate set called cross validation set is prepeared
# from dataset( other than training set and test set). Than in this set you experiment different parameters of your model.This will be done later.

#input_dim is equal to 11 i.e. number of nodes in input layer is equal to number of independent variable.

#here init means initializing weights closer to zero which is done by uniform.

#relu is the short form for rectifier activation function.
# input_dim is for number of input nodes in input layer.
#hit ctrl+enter


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#here we won't need to add input_dim because our first hidden layer is already created.
#initialize the weight that comes from first layer.
#hit ctrl+enter

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#here we have one output node, therefore 1 is used. Moreove, since we have 
#binary outcome i.e. either perosn stays with bank or leaves the bank.
#initialize the weight that comes from second layer.
#since this is an output layer, thus we have used sigmoid function which is the heart of this model
#because it  gives the probability

#if we had two categories of dependent variable than:
#in above output_dim =2 and activation ='sigmoid'

#if we had three or four or more categories of dependent variable than(very important):
#output_dim = 3 , activation ='soft max'

#hit ctrl +enter


# Compiling the ANN
#compiling means applying stochastic gradient.
 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#now press ctrl +I on compile and you will see three arguments.
#optimizer is our algorithm which is used to decide the optimal set of weights in neural network
#because till now we have just initialized our weights but now optimizer will optimize them.
#hence we need to give stochastic gradient to optimizer which is known as Adam.

#here loss function is optimized via stochastic gradient to eventually find optimal weights.
#in linear regression loss is the sum of square of the difference between output value and actual value.
#and in logistic regression it is the logarithmic loss. 
#and here in ANN our loss function is also logarithmic loss of logistic regression and
#not of linear regression.
#if our dependent variable has binary outcome than this logarithmic loss is called binary_crossentropy
# and if our dependent variable has more than two outcome than this logarithmic loss is called categorical_crossentropy.
#and since here our dependent variable has two possible outcome therefore we have binary_crossentropy.

#third argument is metrics:
#after each observation or a batch of observation the weights are updated, the algorithm 
#uses this accuracy criterion to improve the model performance.
#This accuracy will be observed when we fit our training set into ANN.
#and since in the 'help' section it is clearly mentioned that metric has to be a list,
#hence we will put accruracy metirics inside list [].
 




# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#one epoch means all observations in a dataset. Here we have 10,000 observations in our epoch(present in our datset).
#out of which 8000 are present in X_train and Y_train.
#batch_size =10 means that in every epoch we will choose 10 datasets at one time and update our weights.
#than next 10 dataset will be used to update weight and this process keeps on occuring until 8,000 observations of train set is used.
#then second epoch is used and same procedure occurs and this process continues until 100 epcohs are covered.
#hit ctrl +enter and than we can see magic in console. 
#once entire process gets over we will get 0.8346 which means accuracy is around 83%.
#hence we can expect 83% of accuracy during our test set.
#thus, our ANN has been trained on trainset. and now we use ANN on test set to make predictions.



# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#hit ctrl +enter

#now there are 50% chances either a person leaves or stays, hence we are keeping the threshold to be 0.5 
y_pred = (y_pred > 0.5)
#thus if y_pred is larger than 0.5 it return true otherwise it returns false.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#now let us check accuracy by calculating it in console:
 