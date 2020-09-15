insurance <-read.csv("C:/Users/USER/Downloads/Insurance Dataset.csv")
install.packages("plyr")
library(plyr)

#selecting k value based on sqrt(n/2) formula
norm_1 <- scale(insurance)# converting entire value in dataframe into z values
km <- kmeans(norm_1,7)
str(km)
km$cluster
install.packages("kselection")
library(kselection)
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores = 6)# parallel =TRUE, decreases the time of execution of command, because 6 cores are used for running it parallel
k <- kselection(norm_1,parallel =TRUE, k_threshold=0.9, max_centers=15)


k

twss= NULL

for(i in 1:12){
  twss[i]=sum(kmeans(norm_1,i+1)$withinss)
}
twss
plot(2:13,twss,type = 'b',xplot='no. of clusters',yplot='twss')#by seeing screeplot we can find the elbow at  9
km<- kmeans(norm_1,9)# considering 9 clusters for this dataset

str(km)
km$cluster
insur <- cbind(insurance,km$cluster)
insur
km$centers
km$iter
km$withinss

