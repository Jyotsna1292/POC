
# import data
company <- read.csv("C:/Users/USER/Downloads/Company_Data.csv")

# sales is the target variable so we will be converting it into categorical form.
# categories are low=(<5), medium=(5 to 10), high=(10 to 15), very high= (>15))
sales_cat <- cut(company$Sales, breaks =c(0,5,10,15,20), labels = c("low","medium","high","very high"),right = FALSE)
sales_cat

any(is.na(company))

comp <- cbind(company,sales_cat)

# splitting data into training and testing

smp_size <- floor(0.75 * nrow(comp))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(comp)), size = smp_size)

train <- comp[train_ind, ]
test <- comp[-train_ind, ]

table(comp$sales_cat)

# Step 3: Training a model on the data
install.packages("C50")
library(C50)
sales_model <- C5.0(train[ ,2:11], train[ ,12])
plot(sales_model)

train_pred <- predict(sales_model, train[ ,2:11])
train_acc <- mean(train[ ,12]==train_pred)
train_acc # 93.66%

# test accuracy
test_pred <- predict(sales_model, test[ ,2:11])
test_acc <- mean(test[ ,12]==test_pred)
test_acc # 61%

# cross tabulation of predicted versus actual classes
install.packages("gmodels")
library(gmodels)
CrossTable(test[ ,12], test_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))






