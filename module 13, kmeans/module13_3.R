crime <- read.csv("C:/Users/USER/Downloads/crime_data.csv")
attach(crime)

crime.data <- crime[,-1]# k value choosen based on sqrt n/2
norm_crime <- scale(crime.data) 
km <- kmeans(norm_crime, 5)
str(km)

#scree plot
TWSS= NULL
for (i in 1:10) {
  TWSS[i]<-sum(kmeans(norm_crime,i+1)$withinss)
  
} 
TWSS

plot(2:11, TWSS,type = 'b', xplot= 'no. of cluster',yplot='TWSS')
# my scree plot is telling me to take k value as 4 or 5

km1 <- kmeans(norm_crime,5)
str(km1)
km1$cluster
crime_data <- cbind(crime,km1$cluster)
crime_data











