# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:18:31 2020

@author: USER
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re
 
interstellar_reviews=[]

url="https://www.imdb.com/title/tt0816692/reviews?ref_=tt_ov_rt"
response = requests.get(url)
soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
reviews = soup.findAll("div",attrs={"class","text show-more__control"})# Extracting the content under specific tags  
for i in range(len(reviews)):
    interstellar_reviews.append(reviews[i].text)  
  
# writng reviews in a text file 
with open("insterstellar.txt","w",encoding='utf8') as output:
    output.write(str(interstellar_reviews))
    
import os
os.getcwd()

# Joinining all the reviews into single paragraph 
in_rev_string = " ".join(interstellar_reviews)



# Removing unwanted symbols incase if exists
in_rev_string = re.sub("[^A-Za-z" "]+"," ",in_rev_string).lower()
in_rev_string = re.sub("[0-9" "]+"," ",in_rev_string)



# words that contained in bluetooth speaker reviews
in_reviews_words = in_rev_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:/Users/USER/Documents/stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

in_reviews_words = [w for w in in_reviews_words if not w in stopwords]



# Joinining all the reviews into single paragraph 
in_rev_string = " ".join(in_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud

pip install wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud


wordcloud_in = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(in_rev_string)

plt.imshow(wordcloud_in)

# positive words # Choose the path for +ve words stored in system
with open("C:/Users/USER/Documents/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
#poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("C:/Users/USER/Documents/negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
in_neg_in_neg = " ".join ([w for w in in_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(in_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
in_pos_in_pos = " ".join ([w for w in in_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(in_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

