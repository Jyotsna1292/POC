
#Installing and loading the libraries
install.packages("recommenderlab", dependencies=TRUE)
install.packages("Matrix")
library("recommenderlab")
library(caTools)
library(Matrix)

#movie rating data
book_data <- read.csv("C:/Users/USER/Downloads/book (1).csv")

#metadata about the variable, metadata is data which provide information about other data
str(book_data)

install.packages("tidyr")
library(tidyr)

book_data %>% separate(Book.Author, 
                c("bookauthor"))

book_data %>% separate(Publisher, 
                       c("publisher"))
# removing 1st two columns
book_data <- book_data[ ,3:6]
#the datatype should be realRatingMatrix inorder to build recommendation engine
book_data_matrix <- as(book_data, 'realRatingMatrix')

#Popularity based 

book_recomm_model1 <- Recommender(book_data_matrix, method="POPULAR")
book_recomm_model1

#Predictions for two users 
recommended_items1 <- predict(book_recomm_model1, book_data_matrix[4839], n=5)
as(recommended_items1, "list")


## Popularity model recommends the same books for all users , we need to improve our model using # # Collaborative Filtering

#User Based Collaborative Filtering

book_recomm_model2 <- Recommender(book_data_matrix, method="UBCF")
book_recomm_model2

#Predictions for two users 
recommended_items2 <- predict(book_recomm_model2, book_data_matrix[2000], n=5)
as(recommended_items2, "list")



