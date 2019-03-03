#see the change if you use 'delete' of keyboard and to nullify it's effects use 'insert' present
#on left of delete.

#to run any file first go to that file via file explorer and then
#open that file. once you are done with coding then hit save option and then run it.
#this will make your file - parent directory and output is displayed in console.

#to see whether you have writtern any particular line correct or not, select the entire 
#line and  press 'ctrl + enter'
                                   
#importing libraries
import numpy as np                 #np is used to bring mathematics in our code
import matplotlib.pyplot as plt    # matplotlib is used for piechart and pyplot is sublibrary
import pandas as pd                #pandas used to import and mangade dataset.

#importing datasets 
dataset=pd.read_csv("Data.csv") #read_csv is a method of pandas which puts Data.csv file
                                #in to dataset(a variable). To check this open 
                                #variable explorer(in View) and  "click" on dataset.
                                #If you want to bring any change in Data.csv file
                                #ike add columns and rows then go to Data.csv file by
                                #opening in libreoffice or ms excel via Files option
                                #and not by using file explorer
                                
                                