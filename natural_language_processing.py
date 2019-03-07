# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#press ctrl+enter above all lines and you will see dataset imported 
#to variable explorer.Moreover click the dataset and see the beautiful organization.
#see machine learning.odt, use explained beautifully.
#delimeter /t will tell pandas that there is a tab between two columns.
#sometimes double quotes("") can cause problems inside the review, therefore,
# we are going to use quoting parameter. This parameter has got certain codes,
#and code to ignore double quotes is 3.



# Cleaning the texts
#import re  
#review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])

#here we are using sub function of ‘re’ which will take three parameters:
# first parameter is: [^a-zA-Z]’       where ^(hat) means that remove all parts except letter from a to z as  					       well as capital letters A-Z
#second parameter is: ‘     ’             it gives space between a-z and A-Z to so that complex words are not 					       formed from reviews. 
#Third parameter is:
#dataset['Review'][0]                  first parameter is applied to this third parameter which gives review 					        present at index 0 i.e.Wow... Loved this place.
#press ctrl+enter and operation occurs where puctuation ... is removed and result is stored inside review and also displayed in variable explorer.


#review = review.lower()

#this will convert the entire review Wow... Loved this place in lower case.
#press ctrl+ enter


#import nltk
#nltk.download('stopwords')

#press ctrl+enter and stopwords list(containing all irrelevant words which can be 
#found in any review)
#list of stopward will be downloaded inside nltk and message is displayed in console. 
#once the stopword list is downloaded again hit ctrl+enter then message of uptodate is displayed
#But point to remember is that stopwords package got downloaded in nltk but 
# we need to import in dataset which will be done later in code.*


#review =review.split()

#press ctrl + entere for above line.




 #part1:moreover,*  here we have imported the stopword inside spyder.
 
#review =[word for word in review if not word in set(stopwords.words('english'))]
 
#here 'word for word'  involves for loop which check for word in review.
#then we have 'if not' condition which will check if any word in review is present which matches
#inside stopword, it if is present than it will be removed from review(like 'this' is removed).
# stopwords.words('english') means that only english words will be checked while matching.
# press ctrl+enter.
#clearly in variable explorer our this word is removed.




#part2:now we will discuss about the stemming process where we keep only the root word
# for eg. from loved we will keep only love and not it's past tense loved, will love or loving
#we carry out all this process because ultimately all these words will be saved in sparse matrix 
# each word has it's column in sparse matrix, therefore reducing extra words will remove extra space from sparse matrix
#Stemming is always applied on a single word and not on the whole list.

#thus to use stem function we will import PorterStemmer class from nltk library and 
#than create object 'ps' from class PorterStemmer.

#so we will use:
 
 #from nltk.stem.porter import PorterStemmer 
 #ps =PorterScammmer()   #object is created
# review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
 
#now i have commented all the code above because in order to apply above stemming process
 # so we need to restart kernel, to restart the entire process, hence I am writing the
#entire code again of cleaning the text.
import re  
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
review =review.split()
ps =PorterStemmer()
review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


#now press ctrl+enter from importing libraries till above line.


#now we will covert our list of review back into our string and this is done by using
#join function.
review = ' '.join(review)   #here we have used ' '.join because in order to keep space betwee wow  love   space. otherwise it will appear as wowloveplace
#press ctrl+ enter 





#Now are going to rewrite the above code for all 1000 reviews, hence we are going to use for loop.
import re  
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []              #this corpus is an emptly which will store 1000 reviews and in NLP corpus means huge amount of text of anything.
for i in range(0, 1000):                      #range is from 0 to 999 i.e.1000
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)       #here we are appending our clean review to our corpus.
    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)