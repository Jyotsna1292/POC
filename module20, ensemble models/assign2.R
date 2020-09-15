wbcd <- read.csv("C:/Users/USER/Downloads/wbcd.csv")

data <- wbcd[ ,2:32]
install.packages("caret")
library(caret)

#Accuracy with single model
inTraininglocal<-createDataPartition(data$diagnosis,p=.75,list = F)
training<-data[inTraininglocal,]
testing<-data[-inTraininglocal,]

install.packages("C50")
library(C50)
model<-C5.0(training$diagnosis~.,data = training[,-1])
pred<-predict.C5.0(model,testing[,-1])
a<-table(testing$diagnosis,pred)
a
sum(diag(a))/sum(a) # accuracy is 95.77%

########Bagging
acc<-c()
for(i in 1:11)
{
  inTraininglocal<-createDataPartition(data$diagnosis,p=.85,list = F)
  training1<-data[inTraininglocal,]
  testing<-data[-inTraininglocal,]
  fittree <- C5.0(training1$diagnosis~., data=training1[,-1])
  pred<-predict.C5.0(fittree,testing[,-1])
  a<-table(testing$diagnosis,pred)
  acc<-c(acc,sum(diag(a))/sum(a))
}
summary(acc)
mean(acc) # accuracy improved to  95.23%

####################### Boosting

#Accuracy with single model with Boosting

inTraininglocal<-createDataPartition(data$diagnosis,p=.75,list = F)
training<-data[inTraininglocal,]
testing<-data[-inTraininglocal,]

model<-C5.0(training$diagnosis~.,data = training[,-1],trials=10)
pred<-predict.C5.0(model,testing[,-1])
a<-table(testing$diagnosis,pred)

sum(diag(a))/sum(a) # accuracy increased to 97.18%

######## Bagging and Boosting
acc<-c()
for(i in 1:11)
{
  
  inTraininglocal<-createDataPartition(data$diagnosis,p=.85,list = F)
  training1<-data[inTraininglocal,]
  testing<-data[-inTraininglocal,]
  
  fittree <- C5.0(training1$diagnosis~., data=training1,trials=10)
  pred<-predict.C5.0(fittree,testing[,-1])
  a<-table(testing$diagnosis,pred)
  
  acc<-c(acc,sum(diag(a))/sum(a))
  
}

summary(acc)
mean(acc) # accuracy 96.75%












