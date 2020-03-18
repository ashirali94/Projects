# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:58:32 2020

@author: Ashir
"""




import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize 
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from autocorrect import Speller
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup 
spell = Speller(lang='en')
stemmer = PorterStemmer()


"""
IMPORTANT NOTE ---- accuracies and Confusion matrices are written at the end of each block of code

Raw Count Bag of words approach gave the best accuracy ==>  Accuracy : 0.72096
                                                            Confusion Matrix[[2733 1354]
                                                                             [ 390 1773]]

This Code contains 3 parts

1- N-grams using CountVectorizer

2- TF-IDF  using CountVectorizer

3- RAW Counts using CountVectorizer

"""


'''1 - N-GRAMS'''


df = pd.read_csv('dataset.tsv', header = 0, delimiter = '\t', quoting=3)

corpus = df['review'].tolist()

y = df.sentiment

def preprocess(review):
    #remove html using BeautifulSoup
    review = BeautifulSoup(review,"html.parser").get_text()
    review = re.sub('[^A-Za-z]', ' ', review)
    review = review.lower().split()
    words = [word for word in review if not word in stop_words]

    
    return(" ".join(words))
        
#enddef 
    
#Getting preprocessed reviews and putting in list
reviews = []    
for i in range(0,df.review.size):
    reviews.append(preprocess(df.review[i]))

#Converting the words into vectors and creating n-gram vectorizer with char_wb -- unigram and bigram
vec = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
#Converting reviews into features
X = vec.fit_transform(reviews).toarray()

feature_names = vec.get_feature_names()

pd.DataFrame(X, columns = feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes 
print("Naive Bayes using N-gram Bag of words Vectorizor")
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

print('accuracy : '+str(accuracy))
print('Confusion Matrix'+str(cm))

'''
accuracy : 0.5536
Confusion Matrix[[2648 2319]
                 [ 471  812]]
'''





'''2- TF-IDF'''
#####TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vec.fit_transform(reviews).toarray()

feature_names = vec.get_feature_names()

pd.DataFrame(X, columns = feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes 
print("Naive Bayes using TF-IDF Vectorizor")
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

print('accuracy : '+str(accuracy))
print('Confusion Matrix'+str(cm))
'''
accuracy : 0.55728
Confusion Matrix[[2542 2201]
                 [ 566  941]]
'''




'''3 - RAW Counts'''
####With RAW COUNTS

    
#Getting preprocessed reviews and putting in list
reviews = []    
for i in range(0,df.review.size):
    reviews.append(preprocess(df.review[i]))

print("Naive Bayes using RAW COUNTS Vectorizor")
vec = CountVectorizer(analyzer = "word",
                      tokenizer = None,
                      preprocessor = None,
                      stop_words = None,
                      max_features = 5000)
#Converting reviews into features
X = vec.fit_transform(reviews).toarray()

feature_names = vec.get_feature_names()

pd.DataFrame(X, columns = feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y)
# Naive Bayes 
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

print('accuracy : '+str(accuracy))
print('Confusion Matrix'+str(cm))

'''
accuracy : 0.72096
Confusion Matrix[[2733 1354]
                 [ 390 1773]]
'''
