
# import dataset
election <- read.csv('C:/Users/USER/Downloads/election_data.csv')

# removing row having null value
Election <- election[2:11,1:5]

attach(Election)

# Preparing a linear regression 
mod_lm <- lm(Result~.,data=Election)
pred1 <- predict(mod_lm,Election)
pred1

# preparing logistic regression model
model <- glm(Result~.,data=Election,family = "binomial")
summary(model)
# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))


# Confusion matrix table 
prob <- predict(model,Election,type="response")
prob
# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(prob>0.5,Election$Result)
confusion
# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 100%

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>0.5,1,0)
yes_no <- ifelse(prob>0.5,"yes","no")

# Creating new column to store the above values
Election[,"prob"] <- prob
Election[,"pred_values"] <- pred_values
Election[,"yes_no"] <- yes_no

table(Election$Result,Election$pred_values)

# our model is giving 100% accuracy

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic
install.packages("ROCR")
library(ROCR)
rocrpred<-prediction(prob,Election$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
str(rocrperf)
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)



















