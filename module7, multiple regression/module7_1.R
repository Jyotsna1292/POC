startup <- read.csv("C:/Users/USER/Downloads/50_Startups.csv")

hist(R.D.Spend)
boxplot(R.D.Spend)

# creating dummies of categorical column
install.packages("mlr")
library(mlr)
library(ParamHelpers)
startups<-createDummyFeatures(startup, cols = "State")
attach(startups)

# finding correlation 
pairs(startups)
cor(startups)


# there is collinearity problem between R.D spend and Marketing.spend
model <- lm(Profit~R.D.Spend+Administration+Marketing.Spend+State.California+State.Florida+State.New.York, data = startups)
model
summary(model) # r^2 is 0.95

# creating model with only administration
model1 <- lm(Profit~Administration)
summary(model1)

# creating model without administration
model2 <- lm(Profit~R.D.Spend+Marketing.Spend)
summary(model2)

# creating model with marketing spend
model3 <- lm(Profit~Marketing.Spend)
summary(model3)

install.packages("GGally")
library(GGally)
ggpairs(startups)

# for partial corelation matrix
install.packages("corpcor")
library(corpcor)
cor2pcor(cor(startups))
cor(startups)


# these plots are showing that 50th observation is causing problem to the model

#deletion diagonastic for identifying influential variable
influence.measures(model)
influenceIndexPlot(model)
influencePlot(model) # they are also poiting to 50th observation

cor(startups)
summary(startups)

# making model without 50th observation
startup_1 <- startups[-c(50), ]

# making model
model_1 <- lm(Profit~R.D.Spend+Administration+Marketing.Spend+State.California+State.Florida+State.New.York, data = startup_1)
summary(model_1) # r^2 value improved to 0.9618

plot(model_1)

influence.measures(model_1)
influenceIndexPlot(model_1)

#variance inflation factor
vif(model_1) #VIF > 10 = collinearity
VIFRD <- lm(R.D.Spend~Administration+Marketing.Spend+State.California+State.Florida+State.New.York)
VIFA <- lm(Administration~R.D.Spend+Marketing.Spend+State.California+State.Florida+State.New.York)
VIFMS <- lm(Marketing.Spend~R.D.Spend+Administration+State.California+State.Florida+State.New.York)
summary(VIFRD)
summary(VIFA)
summary(VIFMS)

#added variabe plot, avplot
avPlots(model_1) # by visualizing this plot it is clear that administration column contribution to the model is minimal

install.packages("MASS")
library(MASS)
stepAIC(model)

# stepAIC suggesting to make model without administration column
# final model
model_2 <- lm(Profit~R.D.Spend+Marketing.Spend+State.California+State.Florida+State.New.York, data = startup_1)
summary(model_2) # r^2 is 0.9616 
avPlots(model_2)

# moving profit column to the front
col_idx <- grep("Profit", names(startup_1))
startup_1 <- startup_1[, c(col_idx, (1:ncol(startup_1))[-col_idx])]
names(startup_1)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(startup_1))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(startup_1)), size = smp_size)

train <- startup_1[train_ind, ]
test <- startup_1[-train_ind, ]

pred=predict(model_2,newdata = test[ ,2:7])
pred
actual=test$Profit
actual
error = actual-pred
error
test.rmse=sqrt(mean(error**2))
test.rmse # 5936.273
train.rmse=sqrt(mean(model_2$residuals^2))
train.rmse # 7405.025
