crime <- read.csv('C:/Users/USER/Downloads/crime_data.csv')
crime_data <- scale(crime[,2:5])
d <- dist(crime_data, method = "euclidean")
x <- hclust(d, method = "complete")
plot(x)
plot(x, hang = -1) # dendrogram plot
#by visualizing the dendrogram we get the k value as 5

groups <- cutree(x,k=6)
crime_1 <- cbind(crime,groups)
final_crime <- crime_1[,c(ncol(crime_1),1:ncol(crime_1)-1)]
write.csv(final_crime , file = "crimedata.csv")
getwd()
