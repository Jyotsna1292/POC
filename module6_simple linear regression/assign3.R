
# importing dataset
emp_data <- read.csv("C:/Users/USER/Downloads/emp_data.csv")
attach(emp_data)

# performing EDA
summary(emp_data)
boxplot(emp_data$S.h)
hist(emp_data$S.h)
boxplot(emp_data$C.r)
hist(emp_data$C.r)
plot(C.r,S.h)

# correlation between input and output
cor(C.r, S.h) # -0.9117

# preparing model
model.1 <-lm(S.h~C.r)
model.1
summary(model.1) # R^2 value is 0.8312

# calculating rmse
rmse <- sqrt(mean(model.1$residuals^2))
rmse # 35.89
# doing prediction
predict(model.1)
confint(model.1, level = 0.98)
predict(model.1, interval = 'confidence')

# taking log of one variable
plot(log(C.r),S.h)
cor(log(C.r),S.h) # -0.9346

model.2 <- lm(S.h~log(C.r))
model.2
summary(model.2) # r^2 value got improved to 0.8735

rmse<- sqrt(mean(model.2$residuals^2))
rmse # rmse reduced to 31.069

# taking square of one variable
plot(C.r^2,S.h)
cor(S.h, C.r^2) #-0.8859, correlation got worsen
model.3 <- lm(sqrt(S.h)~C.r)
model.3
summary(model.3) # r^ 2 value dipped to 0.84

 # model.2 is the best model

