# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:27:51 2020

@author: USER
"""

import pandas as pd

# importing dataset
book = pd.read_csv("C:/Users/USER/Downloads/book (1).csv", encoding='ISO-8859-1')

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with
# empty string
book.isnull().sum() # no NaN value present

# we will find similarity based on author name and publisher, so will merge the content of both columns into single column

book.columns
# renaming column names
book = book.rename(columns={'Book.Title': 'booktitle', 'Book.Author': 'bookauthor','ratings[, 3]':'ratings'})

book['Publisher'].value_counts()

book['bookauthor'] = book['bookauthor'].map(lambda x: x.split(';')[0])
book['Publisher'] = book['Publisher'].map(lambda x: x.split(';')[0])

book['bookauth_pub']=book['bookauthor']+' '+book['Publisher']

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(book.bookauth_pub)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000,4512

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of booktitle  to index number 
book_index = pd.Series(book.index,index=book['booktitle']).drop_duplicates()


book_index["Clara Callan"]

def get_book_recommendations(Name,topN):
    
   
    #topN = 10
    # Getting the book index using its title 
    book_id = book_index[Name]
    
    # Getting the pair wise similarity score for all the book's with that 
    # book
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar anime's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the anime index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar books and scores
    book_similar_show = pd.DataFrame(columns=["booktitle","Score"])
    book_similar_show["booktitle"] = book.loc[book_idx,"booktitle"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    #book_similar_show.drop(["index"],axis=1,inplace=True)
    print (book_similar_show)
    #return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_book_recommendations("Clara Callan",topN=15)






















