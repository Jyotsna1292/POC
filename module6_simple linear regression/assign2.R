# importing dataset
d_t <- read.csv("C:/Users/USER/Downloads/delivery_time.csv")
attach(d_t)

# performing EDA
summary(d_t)
plot(S_t, D_t)

# correlation between input and output
cor(D_t, S_t) # 0.82599

# preparing model
model.1 <- lm(D_t~S_t)
model.1
summary(model.1) # R^2 is 0.6823

model.1$residuals
predict(model.1)
confint(model.1, level = 0.95)

# calculating rmse of this model
rmse <- sqrt(mean(model.1$residuals^2))
rmse # 2.79165
predict(model.1, interval = 'confidence')

# taking log of one variable
plot(log(S_t),D_t)
cor(log(S_t),D_t) # 0.8339

model.2 <- lm(D_t~log(S_t))
model.2
summary(model.2) # r^2 is 0.6954
# model has improved to some level

rmse<- sqrt(mean(model.2$residuals^2))
rmse # 2.7331, rmse also reduced

# now taking log of another variable
plot(S_t, log(D_t))
cor(log(D_t),S_t) # 0.843177

model.3 <- lm(log(D_t)~S_t)
model.3
summary(model.3) # r^2 value got further improved to 0.7109
x <-predict(model.3, interval = 'confidence')
x
dt <- exp(x)
dt
err<- D_t-dt
err
rmse <- sqrt(mean(err^2))
rmse # rmse slightly increased to 3.377184
# model.3 will be our final model as it is the best model
