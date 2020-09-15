
# importing dataset
computer <- read.csv("C:/Users/USER/Downloads/Computer_Data.csv")

# dataset consists of several categorical columns, converting all of them into dummy variable
# creating dummies of categorical column
install.packages("dummies")
library(dummies)

computer_new <- dummy.data.frame(computer, sep = ".")
Computer <- computer_new[ ,2:14]

# finding correlation 
pairs(computer)
cor(computer)

# preparing model taking all inputs
model <- lm(price~., data = Computer)
summary(model) # r^2 is 0.7756, all probilities are significant

#diagonastic plots
install.packages("car")
library(car)
library(carData)
plot(model) # it is poiting to two records 1441, 1701 as a problem

#deletion diagonastic for identifying influential variable
influence.measures(model)
influenceIndexPlot(model)
influencePlot(model)

# making model without these observations(1441,1701)
computer_1 <- Computer[-c(1441,1701), ]

# preparing model
model_1 <- lm(price~., data = computer_1)
summary(model_1) # r^2 improved to 77.77

#added variabe plot, avplot
avPlots(model_1) # all columns are contributing to the model

## 75% of the sample size
smp_size <- floor(0.75 * nrow(computer_1))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(computer_1)), size = smp_size)

train <- computer_1[train_ind, ]
test <- computer_1[-train_ind, ]

pred=predict(model_1,newdata = test[ ,2:13])
pred
actual=test$price
actual
error = actual-pred
error
test.rmse=sqrt(mean(error**2))
test.rmse # 271.9864
train.rmse=sqrt(mean(model_1$residuals^2))
train.rmse # 272.8675







