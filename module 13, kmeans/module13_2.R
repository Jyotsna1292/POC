
# importing dataset
install.packages("readxl")
library(readxl)

airlines <- read_("C:/Users/USER/Downloads/Airlines.xlsx")
attach(airlines)

#airlines.data <- airlines[,-1]# k value choosen based on sqrt n/2
norm_airlines <- scale(airlines) 
km <- kmeans(norm_airlines, 5)
str(km)

#scree plot
TWSS= NULL
for (i in 1:30) {
  TWSS[i]<-sum(kmeans(norm_airlines,i+1)$withinss)
  
} 
TWSS

plot(2:31, TWSS,type = 'b', xplot= 'no. of cluster',yplot='TWSS')
# my scree plot is telling me to take k value as 4 or 5

km1 <- kmeans(norm_airlines,12)
str(km1)
km1$cluster
airlines_data <- cbind(airlines,km1$cluster)
airlines_data











