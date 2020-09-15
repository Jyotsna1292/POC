
# importing dataset
salary <- read.csv("C:/Users/USER/Downloads/Salary_Data.csv")
attach(salary)

# performing EDA on the dataset
summary(salary)
boxplot(salary$ex)
hist(salary$ex)
boxplot(salary$S)
hist(salary$S)

# finding correlation between input and output
plot(ex,S)
cor(ex,S) # 0.97, very strong positive correlation exists

# preparing model
model.1<- lm(S~ex)
model.1
summary(model.1) # R^2 value is extremely high = 0.957
rmse<- sqrt(mean(model.1$residuals^2))
rmse # 5592.044


confint(model.1, level = 0.95)
predict(model.1, interval = 'confidence')

# taking log of input
plot(log(ex),S)
cor(log(ex),S) # 0.924

model.2 <- lm(S~log(ex))
model.2
summary(model.2) # r^2 is 0.8539, model got worsen

# taking sqrt of input
plot(sqrt(ex),S)
cor(sqrt(ex),S) # 0.964

model.3 <- lm(S~sqrt(ex))
model.3 # r^2 value is 0.931
summary(model.3)

# model.1 is the best model
