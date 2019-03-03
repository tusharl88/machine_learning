# see machine learning odt to learn about the following code.
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')   #usin pandas library

X = dataset.iloc[:, :-1].values     #select the entire line and click ctrl+ enter, we will see that see that statement got 
#got executed inside console as well as dataset X is formed inside variable explorer.
#x is the matrix of independent variable # i.e. years of experience in dataset
#here -1 means that we have removed salary, the last column and consider only year of experience. click on X in variable 
#explorer and you will see only year of experience and not salary.
#in front of X we can see size is (30,1) which means 30 options in one column because it is matrix


y = dataset.iloc[:, 1].values   #same process needs to be done here. here 1 means that we are considering
# ony salary and not years of experience.
#in front of y we can see size is (30,) which means it is vector.

#difference between [:,:-1] it means we are creating a 2D object, that's why X is matrix here
#whereas [:,1] means 1D object, that's why y is vector here.


# Splitting the dataset into the Training set and Test set
#X(containing 30 observations) is split into training set(containing 20 of 30 observations) and test set(containing 10 of 30 observations) 
#similarily y consist of training set and test set. hence total four datasets are formed.

#so select the below two lines and click ctrl + enter and four dataset are formed in variable explorer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#clearly here test size is 1/3 which means 10 observations.
#Step1:moreove X training set and Y training set are co related and our training model is prepared.
#Step2:model can later predict salaries based on experience which it will predict by using test cases which 
#are present in test set of x. 
#step3:then we will get the value of predicted salary which we will compare with 
#atual salary present in Y test set.
#this entire process is known as data preprocessing.


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#in simple linear regression we don't use Feature scaling, that's why it has been commented out by using """   """


# Fitting Simple Linear Regression to the Training set
#select linearRegression class and press ctrl+I which displays help in R.H.S which gives information about linear regression class.
from sklearn.linear_model import LinearRegression  #here we are importing library from sklearn (which is itself a library and consist of many libraries) for implementing linear regression.
regressor = LinearRegression()      #we are using linear_model library from where we get linearRegreesion in-built class which consist of fit method. 
regressor.fit(X_train, y_train)    #here regressor is the object of class linear regression
                                   #select fit and click ctrl+I and you will get the information about fit method on R.H.S
#finally select the entire code and ctrl+enter and we will see that regressor is created and fitted to the training datatset
#Note: so finally here our model linear regression learns from the available dataset which we proveided and finally going to use it to prdict.
                                   


# Predicting the Test set results
# here the model which we have prepared(y using training set) is used to find the predicted salaries                                
y_pred = regressor.predict(X_test)   #predict method is of linearregression class which will give the predicted salary by using test set X and the predicted salary_y is compared with test set_y
#now select the above line and press ctrl + enter. this will give y_predict in variable explore. Click it and also y_test and compare them.


# Visualising the Training set results
#so here we are using matplot library. 
plt.scatter(X_train, y_train, color = 'red')     #this will show points scattered around regression line. here X_train is going to be x-cordinate and y_train is going to be y-coordinate.
plt.plot(X_train, regressor.predict(X_train), color = 'blue')   #this will give regression line
#in the above line x-coordinate is X_train and Y-coordinate is  regressor.predict(X_train). here we have used X_train and y_train because we have to predict the y_train using x_train.
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#highlight the entire code from plt.scatter to plt.show() and press ctrl+enter. You will get the entire result. 
#now we can easily see that some points lie closer to the regression line and some are far.
#Note: so all the points whcih are red in color are the observations of Training set(not test set) 
#and our linear regression model was trained on these red points.


# Visualising the Test set results
#now here we will use our trained regression model on test set observations.
#we will plot same blue line but this time we will have new observations in orange color.
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #very important to note here is that we will rain  X_t ,not X_test and y-test because we need use regression model which we trained using red points and not orange points.
#moreover by using regressonr.predict we were able to predict salary and generate orange points.
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()