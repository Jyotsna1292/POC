
# importing dataset
calories_consumed<-read.csv("C:/Users/USER/Downloads/calories_consumed.csv")
View(calories_consumed)
attach(calories_consumed)

# performing EDA on dataset
summary(calories_consumed)
hist(calories_consumed$Weight.gained..grams.)
plot(calories_consumed$Calories.Consumed, calories_consumed$Weight.gained..grams.) #for plotting 
boxplot(calories_consumed$Weight.gained..grams.)
boxplot(calories_consumed$Calories.Consumed)

# finding correlation between input and output
cor(calories_consumed$Weight.gained..grams.,calories_consumed$Calories.Consumed) # 0.946

# preparing model
model.1 <- lm(Weight.gained..grams.~Calories.Consumed) # for linear regression
model.1
summary(model.1)
model.1$residuals
predict(model.1)

# calculating rmse
rmse <- sqrt(mean(model.1$residuals^2))
rmse # 103.3025

# transforming input
plot(sqrt(Calories.Consumed),Weight.gained..grams., data="calories_consumed")
cor(Weight.gained..grams., sqrt(Calories.Consumed)) # 0.9255

model.2 <- lm(Weight.gained..grams.~sqrt(Calories.Consumed))
model.2 # r^2 is 0.8567
summary(model.2)
rmse2 <- sqrt(mean(model.2$residuals^2))
rmse2
model.2$residuals # 121.7122
predict(model.2)
confint(model.1, level = 0.95)
# model is getting worse

# taking log of output
plot(log(Calories.Consumed),Weight.gained..grams., data="calories_consumed")
cor(Weight.gained..grams., log(Calories.Consumed)) # 0.898, correlation becomes worse

# we are getting best model without any transformation





