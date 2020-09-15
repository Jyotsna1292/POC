
#Installing and loading the libraries
install.packages("recommenderlab", dependencies=TRUE)
install.packages("Matrix")
library("recommenderlab")
library(caTools)
library(Matrix)


#movie rating data
movie_rate_data <- read.csv("C:/Users/USER/Downloads/Movie.csv")

#metadata about the variable, metadata is data which provide information about other data
str(movie_rate_data)


#rating distribution
hist(movie_rate_data$rating)

#the datatype should be realRatingMatrix inorder to build recommendation engine
movie_rate_data_matrix <- as(movie_rate_data, 'realRatingMatrix')

#Popularity based 

movie_recomm_model1 <- Recommender(movie_rate_data_matrix, method="POPULAR")
movie_recomm_model1

#Predictions for two users 
recommended_items1 <- predict(movie_recomm_model1, movie_rate_data_matrix[4081], n=5)
as(recommended_items1, "list")


## Popularity model recommends the same movies for all users , we need to improve our model using # # Collaborative Filtering

#User Based Collaborative Filtering

movie_recomm_model2 <- Recommender(movie_rate_data_matrix, method="UBCF")
movie_recomm_model2

#Predictions for two users 
recommended_items2 <- predict(movie_recomm_model2, movie_rate_data_matrix[4081], n=5)
as(recommended_items2, "list")


