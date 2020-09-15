#import datasets


salarydata_train <- read.csv("C:/Users/USER/Downloads/SalaryData_Train.csv")
salarydata_test <- read.csv("C:/Users/USER/Downloads/SalaryData_Test.csv")

str(salarydata_train)

salarydata_train_in <- salarydata_train[,1:13]
salarydata_train_out <- salarydata_train[,14]
salarydata_test_in <- salarydata_test[,1:13]
salarydata_test_out <- salarydata_test[,14]

prop.table(table(salarydata_test_out))
prop.table(table(salarydata_train_out))

##  Training a model on the data ----
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(salarydata_train_in, salarydata_train_out)
sms_classifier

##  Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, salarydata_test_in)

table(sms_test_pred)
prop.table(table(sms_test_pred))

install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, salarydata_test_out,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
accuracy <- mean(sms_test_pred==salarydata_test_out)*100 # 81.93%





