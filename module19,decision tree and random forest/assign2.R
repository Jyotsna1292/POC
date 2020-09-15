
# Read the dataset
diabetes <- read.csv("C:/Users/USER/Downloads/Diabetes.csv")

table(diabetes$Class.variable)

round(prop.table(table(diabetes$Class.variable))*100,1)

#Create a function to normalize the data
norm <- function(x){ 
  return((x-min(x))/(max(x)-min(x)))
}

#Apply the normalization function to diabetes dataset
diabetes_norm <-as.data.frame(lapply(diabetes[1:8],norm))

# spliting dataset into train and test
train_x <- diabetes_norm[1:550, ]
train_y <- diabetes[1:550,9]

test_x <- diabetes_norm[550:768, ]
test_y <- diabetes[550:768,9]

# Building a random forest model on training data 
install.packages("randomForest")
library(randomForest)

diabetes_forest <- randomForest(train_y~.,data=train_x,importance=TRUE)
plot(diabetes_forest)

# Train Data Accuracy
train_acc <- mean(train_y==predict(diabetes_forest))
train_acc # 75%

# Test Data Accuracy
test_acc <- mean(test_y==predict(diabetes_forest, newdata=test_x))
test_acc # 81.7%

varImpPlot(diabetes_forest)














