# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:20:29 2020

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# importing dataset
email_data = pd.read_csv("C:/Users/USER/Downloads/sms_raw_NB.csv", encoding='latin-1')

# cleaning data
import re
stop_words = []
with open("C:/Users/USER/Documents/stopwords_en.txt",'r') as f: # importing text file
    stop_words= f.read()

# splitting the entire string by giving separator as "\n" to get list of all stop words
stop_words = stop_words.split("\n") 

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))
     
email_data.text = email_data.text.apply(cleaning_text)    

# removing empty rows
email_data = email_data.loc[email_data.text != " ",:]

# count vectorizer , convert collection of text documents to a matrix of taken counts
# tfidf transformers, transform a count matrix to normalized tf or tf-idf representation

# creating matrix of token count for entire text document
def split_into_words(i):
    return[word for word in i.split(" ")]
    
# splitting data into train and test
from sklearn.model_selection import train_test_split    

email_train,email_test = train_test_split(email_data, test_size=0.3)
                                          
# preparing email text into word count maatrix format
email_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)

# for all messages 
all_email_matrix = email_bow.transform(email_data.text)

# for training messages 
train_email_matrix = email_bow.transform(email_train.text)

# for test messages
test_email_matrix = email_bow.transform(email_test.text)

# learning term weighting  and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_email_matrix)

# preparing tfidf for train emails
train_tfidf = tfidf_transformer.transform(train_email_matrix)
train_tfidf.shape

# preparing tfidf for test emails
test_tfidf = tfidf_transformer.transform(test_email_matrix)
test_tfidf.shape

# preparing naive bayes model on training data test
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# multinomial naive bayes
classifier_mb =MB()
classifier_mb.fit(train_tfidf, email_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m== email_train.type) # 96%

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 95%

# Gaussian Naive bayes
classifier_gb= GB()
classifier_gb.fit(train_tfidf.toarray(), email_train.type.values)
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type) # 90.56%

test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g == email_test.type) # 84.47%












   
