
# importing dataset

Wine<-read.csv("C:/Users/USER/Downloads/wine.csv")
wine<-Wine[,2:14]
attach(wine)

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

#kmeans clustering using PCA 

twss= NULL

for(i in 1:10){
  twss[i]=sum(kmeans(wine_pca,i+1)$withinss)
}
twss
plot(2:11,twss,type = 'b',xplot='no. of clusters',yplot='twss')#by seeing screeplot we can find the elbow at 3

# taking k =3
model<-kmeans(final,3) 
str(model)

cluster<-model$cluster
final_df<-cbind(wine_pca,cluster)

count <- table(final_df$cluster)
count

# kmeans clustering without PCA
wine_1 = scale(wine)

twss= NULL

for(i in 1:12){
  twss[i]=sum(kmeans(wine_1,i+1)$withinss)
}
twss
plot(2:13,twss,type = 'b',xplot='no. of clusters',yplot='twss')

#by seeing screeplot we can find the elbow at 3
model_1<- kmeans(wine_1,3)
model_1$cluster

new_df <- cbind(wine,model_1$cluster)
count_1 <- table(new_df$`model_1$cluster`)
count_1
# we are getting almost same kmeans clustering with and without PCA











