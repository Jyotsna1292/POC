
# importing dataset
forestfire <- read.csv("C:/Users/USER/Downloads/forestfires.csv")

# removing undesirable columns
forest <- forestfire[ ,3:31]

# splitting data into train and test

## 75% of the sample size
smp_size <- floor(0.75 * nrow(forest))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(forest)), size = smp_size)

train <- forest[train_ind, ]
test <- forest[-train_ind, ]

##Training a model on the data ----
# begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)
forest_classifier <- ksvm(size_category ~ ., data = train,
                          kernel = "vanilladot")
forest_classifier

forest_predictions <- predict(forest_classifier, test)
head(forest_predictions)

table(forest_predictions,test$size_category)

agreement <- forest_predictions == test$size_category
table(agreement)
prop.table(table(agreement)) # accuracy is 98.46%

## using kernal rbfdot ----
forest_classifier_rbf <- ksvm(size_category ~ ., data = train, kernel = "rbfdot")
forest_predictions_rbf <- predict(forest_classifier_rbf,test)

agreement_rbf <- forest_predictions_rbf == test$size_category
table(agreement_rbf)
prop.table(table(agreement_rbf)) # accuracy is 84.6%, accuracy got reduced drastically 

# we are getting very good accuracy by using kernal= vanilladot, so will consider that model for classification







