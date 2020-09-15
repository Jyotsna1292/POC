Wine<-read.csv("C:/Users/USER/Downloads/wine.csv")
wine<-Wine[,2:14]
attach(wine)

pcaobj<-princomp(wine,cor=TRUE, scores = TRUE,covmat = NULL) # applying PCA
str(pcaobj)
loadings(pcaobj)
plot(pcaobj)
summary(pcaobj)
pcaobj$sdev*pcaobj$sdev
pcaobj$scores
plot(cumsum(pcaobj$sdev*pcaobj$sdev)*100/sum(pcaobj$sdev*pcaobj$sdev), type = "b")
final<-pcaobj$scores[,1:3]

#kmeans clustering
twss= NULL

for(i in 1:15){
  twss[i]=sum(kmeans(final,i+1)$withinss)
}
twss
plot(2:16,twss,type = 'b',xplot='no. of clusters',yplot='twss')#by seeing screeplot we can find the elbow at  9

# taking k =9
model<-kmeans(final,9) # getting between.ss=1306 as highest, cluster ratio is 0.85
str(model)
cluster<-model$cluster
final_df<-cbind(wine,cluster)





