
# importing dataset
Wine<-read.csv("C:/Users/USER/Downloads/wine.csv")
wine<-Wine[,2:14]
attach(wine)

# performing heirerichal clustering without PCA

d <- dist(wine, method = "euclidean")
x <- hclust(d, method = "complete")
plot(x)
plot(x, hang = -1) # dendrogram plot
#by visualizing the dendrogram we get the k value as 3

groups <- cutree(x,k=3)
wine_1 <- cbind(wine,groups)

count <- table(wine_1$groups)
count

# performing heirerichal clustering with PCA
# applying PCA on the dataset
pcaobj<-princomp(wine,cor=TRUE, scores = TRUE,covmat = NULL) 
str(pcaobj)
loadings(pcaobj)
plot(pcaobj)
summary(pcaobj)
pcaobj$sdev*pcaobj$sdev
pcaobj$scores

plot(cumsum(pcaobj$sdev*pcaobj$sdev)*100/sum(pcaobj$sdev*pcaobj$sdev), type = "b")

wine_pca <- pcaobj$scores[ ,1:3]

d <- dist(wine_pca, method = "euclidean")
x <- hclust(d, method = "complete")
plot(x)
plot(x, hang = -1) # dendrogram plot
#by visualizing the dendrogram we get the k value as 4

groups <- cutree(x,k=4)
wine_2 <- cbind(wine,groups)

count_1 <- table(wine_2$groups)
count_1

# we are getting 4 clusters with PCA and 3 clusters without PCA





