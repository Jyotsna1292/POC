
install.packages('AER')
data(Affairs,package="AER")
attach(Affairs)

# converting all the values in nffairs column which are 1 or more than 1 into 1
Affairs$newaffairs <- ifelse(affairs==0,0,1)

Affairs = Affairs[ ,2:10]

install.packages("fastDummies")
library(fastDummies)

# creating dummies of categorical inputs
Affairs_new <- dummy_cols(Affairs, select_columns=c("gender","children"))

Affairs_new <- subset(Affairs_new, select = -c(gender,children))

# preparing logistic regression model
model <- glm(newaffairs~.,data=Affairs_new,family = "binomial")
summary(model)

# Confusion matrix table 
prob <- predict(model,Affairs_new,type="response")
prob

# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(prob>0.5,Affairs_new$newaffairs)
confusion
# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 76.53%

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>0.5,1,0)
yes_no <- ifelse(prob>0.5,"yes","no")

# Creating new column to store the above values
Affairs_new[,"prob"] <- prob
Affairs_new[,"pred_values"] <- pred_values
Affairs_new[,"yes_no"] <- yes_no

table(Affairs_new$newaffairs,Affairs_new$pred_values)

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic
install.packages("ROCR")
library(ROCR)
rocrpred<-prediction(prob,Affairs_new$newaffairs)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
rocrperf
## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
str(rocrperf)
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

install.packages("dplyr")
library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)

# after visualizing the ROCR plot it is clear that best cutoff value will be 0.47, so will make model with cutoff 0.8


# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(prob>0.47,Affairs_new$newaffairs)
confusion
# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 77.53%











