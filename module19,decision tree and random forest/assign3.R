
# import dataset
fraud <- read.csv("C:/Users/USER/Downloads/Fraud_check.csv")
names(fraud)

# shifting taxable.income variable to front
col_idx <- grep("Taxable.Income", names(fraud))
Fraud <- fraud[, c(col_idx, (1:ncol(fraud))[-col_idx])]

# converting continous data into category 'risky and good'
fraud_check <- cut(Fraud$Taxable.Income, breaks =c(10000,30000,100000),labels=c("Risky","Good"),right=TRUE)
fraud_data <- cbind(Fraud,fraud_check)

# splitting data into training and testing

smp_size <- floor(0.75 * nrow(fraud_data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(fraud_data)), size = smp_size)

train <- fraud_data[train_ind, ]
test <- fraud_data[-train_ind, ]

table(fraud_data$fraud_check)

# Step 3: Training a model on the data
install.packages("C50")
library(C50)
sales_model <- C5.0(train[ ,2:6], train[ ,7])
plot(sales_model)

train_pred <- predict(sales_model, train[ ,2:6])
train_acc <- mean(train[ ,7]==train_pred)
train_acc # 80.22%

# test accuracy
test_pred <- predict(sales_model, test[ ,2:6])
test_acc <- mean(test[ ,7]==test_pred)
test_acc # 76%

# cross tabulation of predicted versus actual classes
install.packages("gmodels")
library(gmodels)
CrossTable(test[ ,7], test_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))














