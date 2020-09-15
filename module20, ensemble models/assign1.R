
diabetes <- read.csv("C:/Users/USER/Downloads/Diabetes_RF.csv")

install.packages("caret")
library(caret)

#Accuracy with single model
inTraininglocal<-createDataPartition(diabetes$Class.variable,p=.75,list = F)
training<-diabetes[inTraininglocal,]
testing<-diabetes[-inTraininglocal,]

install.packages("C50")
library(C50)
model<-C5.0(training$Class.variable~.,data = training[,-9])
pred<-predict.C5.0(model,testing[,-9])
a<-table(testing$Class.variable,pred)
a
sum(diag(a))/sum(a) # accuracy is 69.79%

########Bagging
acc<-c()
for(i in 1:11)
{
  inTraininglocal<-createDataPartition(diabetes$Class.variable,p=.85,list = F)
  training1<-diabetes[inTraininglocal,]
  testing<-diabetes[-inTraininglocal,]
  fittree <- C5.0(training1$Class.variable~., data=training1[,-9])
  pred<-predict.C5.0(fittree,testing[,-9])
  a<-table(testing$Class.variable,pred)
  acc<-c(acc,sum(diag(a))/sum(a))
}
summary(acc)
mean(acc) # accuracy improved to  75%

####################### Boosting


#Accuracy with single model with Boosting

inTraininglocal<-createDataPartition(diabetes$Class.variable,p=.75,list = F)
training<-diabetes[inTraininglocal,]
testing<-diabetes[-inTraininglocal,]

model<-C5.0(training$Class.variable~.,data = training[,-9],trials=10)
pred<-predict.C5.0(model,testing[,-9])
a<-table(testing$Class.variable,pred)

sum(diag(a))/sum(a) # accuracy increased to 79.68%

######## Bagging and Boosting
acc<-c()
for(i in 1:11)
{
  
  inTraininglocal<-createDataPartition(diabetes$Class.variable,p=.85,list = F)
  training1<-diabetes[inTraininglocal,]
  testing<-diabetes[-inTraininglocal,]
  
  fittree <- C5.0(training1$Class.variable~., data=training1,trials=10)
  pred<-predict.C5.0(fittree,testing[,-9])
  a<-table(testing$Class.variable,pred)
  
  acc<-c(acc,sum(diag(a))/sum(a))
  
}

summary(acc)
mean(acc) # accuracy 73.83%












