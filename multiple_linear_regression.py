# Multiple Linear Regression
#whenever you open this file, just select a block of code upto ctrl+enter line.
# And run it ALL OVER AGAIN- EACH BLOCK(many changes occur in dataset during the process).
#you have already left gap for between each block. So run the block separately.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#select above three lines and press ctrl+ enter to get result in console.





# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values # it is the matrix which includes only independent variables 
                                #and excluding profit i.e. dependent variable. 
                                #That's why we have used -1 which will exclude the last column of profit(open the dataset after running these three lines)
y = dataset.iloc[:, 4].values  #it includes the the fourth column of the dataset ONLY which is 
# profit i.e. dependent variable. open the dataset from variable explorer and count from  R&D spend(1 means 0th column)
#till profit(4 means 5th column). Don't consider index.
#Note:1. clearly in variable explorer we can see that X  (50,4)is  two dimensional matrix with 50 rows and 4 colums.
#moreover y, is vector(50,) which is one-dimensional.
#2. Here on R.H.S in variable explorer we can easily see X but when we click it, gives error of array 
#editor. This is because it is of type 'object' and not flot. therfore we type X in console and hit enter.
#and entire X is diplayed in console.
#however on other hand if we click on Y then we don't see error because it is of type float64 and
#box appears
#select above three lines and press ctrl+ enter to get result in console. 
#and then only dataset will be displayed inside variable explorer.






# Encoding categorical data
#here as we can see in console that we have got extra column of categorical variable i.e. of new york,etc.
# Thus we need to Encode this part into numbers and for that we need label encoder.
#And to remove any relational order we need OneHotEncoder to create Dummy variables.
#below code is taken from Part1- Data preprocessing, categorical data.py-----so open that file from file explorer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])    #here [:,3] means colon is for all rows and
                                                 # 3 is for considering only column 4th of State(new york, california) which needs to be to be converted into number 
                                                 #i.e.text needs to be converted into number.
                                             #here we have started form column 0 whcih is of R&D
#now onehotencoder will take number and convert into dummy variable
onehotencoder = OneHotEncoder(categorical_features = [3])  #3 is for column 4th.
X = onehotencoder.fit_transform(X).toarray()
#select above code  and press ctrl+enter and than you will see that result is displayed in console.
#moreover go to variable explorer and you will see that type of X has changed from object to float 64..


# Encoding the Dependent Variable- we will NOT use this part because 'y' is our dependent variable profit which is already given in number.
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)







# Avoiding the Dummy Variable Trap(very important line)
#you have explained this concept briliantly inside machinelearning odt see there.
#here you have re
X = X[:,1:] #: is for considering all 50 rows
            #1: is for considering column1 till end i.e. you have removed column 0 of califrornia 
            #and considered column 1 of florida till end. so X now have 50 rows and six columns.

#this thing can be done automatically by the librares and here we just showing.






# Splitting the dataset into the Training set and Test set
#here we will divide out dataset into train set and test set like we did in linear regression
#here test size is 0.2, so train set will have 40 observatios(80%) and test set will have 10 observations(20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#ctrl+ enter above lines and we get result in the variable explorer.
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""







# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression   #this time linear regression is with several independent variables.
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#now since we have many independent variables here (five) thus we need five dimensions which will be difficult, hence we will not plot the graph for it.
#hit ctrl+enterfor above two lines.





# Predicting the Test set results
y_pred = regressor.predict(X_test)
#here we will get prediction by using predict method on X_test which will be stroed in y_pred.
#and y_pred will be compared with y-test to see how well was our prediction.
#press ctrl+ enter above single line and you will see that y_pred appears in variable explorer 
#and you can compare it with y_test.





#building the optimal model using Backward elimination.
#see machine learning odt page 36 and 37 before going throught it.
#here we are using statsmodels library, earlier we were using LINEAR MODEL Library.
import statsmodels.formula.api as sm
X= np.append(arr= np.ones((50,1)).astype(int)    ,values= X,   axis =1)
             #now press ctrl+ enter
             
             
             
             
             
             
             
#let us deal with backward elimiantion i.e. here we decide which variable is important.
#now we will create a matrix which will store our X dataset and all it’s independent variables
#(including x0) . 
X_opt= X[:, [0,1,2,3,4,5]] 
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()  

#we are considering this part inside array [0,1,2,3,4]
#because than we have to remove non-statistically independent variablesi.e. whose value is above p-value
#and that we will be do by using indexes and indexes can only be made inside array.  
#OLS class is present in statsmodels library so we have used sm.
#fit is used for using ordered square method i.e. multiple regression model.
#select above two lines and enter ctrl+enter.
#sometimes you WILL ENCOUNTER ERROR IN CONSOLE THEN IN THAT CASE YOU NEED TO RUN JUST 
#CODE AND NOT COMMENTS so in above code you just have to run only two lines.Don't involve any comment while running above code.



 regressor_OLS.summary()
 #press ctrl+enter
 #here we use summary() function which helps in finding p-value. 
 #see page 41 of machine learning.odt
 
 
 
 #here x2 variable is removed because it has highest value of0.99
 #and corresponding index in x_opt  was 2, that's why
 #2 is not mentioned in below syntax.
 X_opt= X[:, [0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit() 
regressor_OLS.summary()
#press ctrl+enter



#now x1 has the highest value of 0.953
#and greater than SL, so we have to remove it

 X_opt= X[:, [0,3,4,5]] 
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit() 
regressor_OLS.summary()
#press ctrl+enter


#now x3 has the highest p-value and greater than SL value:
 X_opt= X[:, [0,3,5]] 
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit() 
regressor_OLS.summary()
#press ctrl+enter





#now let us remove x2 with value 0.6
 X_opt= X[:, [0,3]] 
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit() 
regressor_OLS.summary()
#press ctrl+enter
#so marketing spend is removed and at last we are left with
#constant and x1 which is R&D spend.
#Therefore, R&D spend is very powerful predictor.

































