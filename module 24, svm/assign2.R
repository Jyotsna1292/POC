# importing dataset

salary_train <- read.csv("C:/Users/USER/Downloads/SalaryData_Train.csv")
salary_test <- read.csv("C:/Users/USER/Downloads/SalaryData_Test.csv")

##Training a model on the data ----
# begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)
salary_classifier <- ksvm(Salary ~ ., data = salary_train,
                          kernel = "vanilladot")
salary_classifier

salary_predictions <- predict(salary_classifier, salary_test)
head(salary_predictions)

table(salary_predictions, salary_test$Salary)

agreement <- salary_predictions == salary_test$Salary
table(agreement)
prop.table(table(agreement)) # accuracy is 84.6%

## Improving model performance ----
salary_classifier_rbf <- ksvm(Salary ~ ., data = salary_train, kernel = "rbfdot")
salary_predictions_rbf <- predict(salary_classifier_rbf, salary_test)

agreement_rbf <- salary_predictions_rbf == salary_test$Salary
table(agreement_rbf)
prop.table(table(agreement_rbf)) # accuracy is 85.41%










