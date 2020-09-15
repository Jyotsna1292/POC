Glass <- read.csv("C:/Users/USER/Downloads/glass.csv") # importing dataset

# checking the structure of data
str(Glass)

# checking frequencies of types
table(Glass$Type)

# checking summary
summary(Glass)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# to check if function working
normalize(c(10, 20, 30, 40, 50))

# normalize the Glass data
glass_n <- as.data.frame(lapply(Glass[1:9], normalize))
type <- Glass[10]

glass <- cbind(glass_n,type)

# creat training and test data
smp_size <- floor(0.85*nrow(glass))
set.seed(123) # for creating same random sample
train_ind <- sample(seq_len(nrow(glass)), size = smp_size)

train <- glass[train_ind, ]
test <- glass[-train_ind, ]


# load the "class" library
install.packages("class")
library(class)

glass_test_pred <- knn(train =train[,1:9], test = test[,1:9],
                      cl = train[,10], k=15) # getting accuracy of 54.54%

test_labels <- test[,10]

# evaluating model performance

# load the "gmodels" library
install.packages("gmodels")
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = test_labels, y = glass_test_pred,
           prop.chisq=FALSE)



test_acc <- mean(glass_test_pred==test[,10])*100
test_acc # accuracy is 48.48%
table(glass_test_pred, test_labels)

glass_test_pred1 <- knn(train =train[,1:9], test = test[,1:9],
                       cl = train[,10], k=14) # getting accuracy of 60.6%
test_acc1 <- mean(glass_test_pred1==test[,10])*100
test_acc1

glass_test_pred2 <- knn(train =train[,1:9], test = test[,1:9],
                        cl = train[,10], k=16) # getting accuracy of 54.54%
test_acc2 <- mean(glass_test_pred2==test[,10])*100
test_acc2

glass_test_pred3 <- knn(train =train[,1:9], test = test[,1:9],
                        cl = train[,10], k=17) # getting accuracy of 54.54%
test_acc3 <- mean(glass_test_pred3==test[,10])*100
test_acc3

glass_test_pred4 <- knn(train =train[,1:9], test = test[,1:9],
                        cl = train[,10], k=18) # getting accuracy of 57.57%
test_acc4 <- mean(glass_test_pred4==test[,10])*100
test_acc4

glass_test_pred5 <- knn(train =train[,1:9], test = test[,1:9],
                        cl = train[,10], k=13) # getting accuracy of 54.54%
test_acc5 <- mean(glass_test_pred5==test[,10])*100
test_acc5

# best accuracy we are getting is 60.60% from this model
######################################################
# improving model performance
# using scale function to standardize data

glass_z <- as.data.frame(scale(Glass[ ,1:9]))
glass_norm <- cbind(glass_z,type)

smp_size <- floor(0.75*nrow(glass_norm))
set.seed(123) # for creating same random sample
train_ind <- sample(seq_len(nrow(glass_norm)), size = smp_size)

train_norm <- glass_norm[train_ind, ]
test_norm <- glass_norm[-train_ind, ]

glass_norm_test_pred <- knn(train =train_norm[,1:9], test = test_norm[,1:9],
                       cl = train_norm[,10], k=21)

# evaluating model performance

test_acc1 <- mean(glass_norm_test_pred==test[,10])*100
test_acc1 # accuracy is 50%

glass_norm_test_pred <- knn(train =train_norm[,1:9], test = test_norm[,1:9],
                            cl = train_norm[,10], k=14)
test_acc1 <- mean(glass_norm_test_pred==test[,10])*100
test_acc1 # accuracy is 46.29%

# we can see that model is not improving after taking scale function so will contine with normalization

#################################################################################

















