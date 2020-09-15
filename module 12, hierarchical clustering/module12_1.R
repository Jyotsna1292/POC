airlines <- read.csv('C:/Users/USER/Downloads/EastwestAirlines.csv')
airlines_data <- scale(airlines[,2:12]) #ignoring ID coloumn
x <- dist(airlines_data, method = "euclidean") #calculating distance matrix
y <- hclust(x , method = "complete")

plot(y) #plotting dendrogram
plot(y, hang = -1)
groups <- cutree(y, k=50)
#rect.hclust(x,k=3, border ='blue')# showing error
cluster <- as.matrix(groups)
final_data <- cbind(airlines,groups)
View(final_data)
final1 <- final_data[,c(ncol(final_data),1:(ncol(final_data)-1))]
write.csv(final1,file = "airlines.csv")
getwd()
