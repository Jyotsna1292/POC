library(readxl)

# model based approach

cocacola <- read_excel("C:/Users/USER/Downloads/CocaCola_Sales_Rawdata.xlsx") # read the cocacola data
View(cocacola) # Seasonality 4 quaters
 
# Pre Processing

# So creating 4 dummy variables 

cocacola["quarter"]=0
x= c('Q1','Q2','Q3','Q4')
t= rep(x, length(cocacola$quarter))
q= t[1:42]
q
cocacola["quarter"]=q

install.packages("dummies")
library(dummies)
x=dummy(cocacola$quarter)
colnames(x) <- c('Q1','Q2','Q3','Q4')

cocacola1 <- cbind(cocacola,x)

# input t
cocacola1["t"] <- c(1:42)
View(cocacola1)

cocacola1["log_Sales"] <- log(cocacola1["Sales"])
cocacola1["t_square"] <- cocacola1["t"]*cocacola1["t"]
View(cocacola1)
## Preprocesing completed

attach(cocacola1)
# partitioning
train <- cocacola1[1:38,]
test <- cocacola1[39:42,]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)
linear_pred <- data.frame(predict(linear_model, interval='predict', newdata =test))
rmse_linear <- sqrt(mean((test$Sales-linear_pred$fit)^2, na.rm = T))
rmse_linear # 591.5533

######################### Exponential #################################

expo_model <- lm(log_Sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval='predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales-exp(expo_pred$fit))^2, na.rm = T))
rmse_expo # 466.248

######################### Quadratic ####################################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)

Quad_pred <- data.frame(predict(Quad_model, interval='predict', newdata=test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm=T))
rmse_Quad # 475.5618

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Q1+Q2+Q3+Q4, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata=test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales-sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add # 1860.024

######################## Additive Seasonality with Quadratic #################

Add_sea_Quad_model <- lm(Sales ~ t+t_square+Q1+Q2+Q3+Q4, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval='predict', newdata=test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad # 301.738

######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Q1+Q2+Q3+Q4, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata=test, interval='predict'))
rmse_multi_sea <- sqrt(mean((test$Sales-exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea # 1963.39

# Preparing table on model and it's RMSE values 

table_rmse <- data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_sea_add","rmse_Add_sea_Quad","rmse_multi_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_sea_add,rmse_Add_sea_Quad,rmse_multi_sea))
colnames(table_rmse) <- c("model","RMSE")
View(table_rmse)

# Additive seasonality with Quadratic has least RMSE value

write.csv(cocacola_1, file="cocacola.csv", row.names = F)

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Sales ~ t+t_square+Q1+Q2+Q3+Q4, data = cocacola1)
summary(Add_sea_Quad_model_final)


####################### Predicting new data #############################

test_data <- read_excel("C:/Users/USER/Downloads/Predict.xlsx")
View(test_data)
pred_new <- predict(Add_sea_Quad_model_final, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)


plot(Add_sea_Quad_model_final)
acf(Add_sea_Quad_model_final$residuals, lag.max = 10) # take all residual value of the model built & plot ACF plot

A <- arima(Add_sea_Quad_model_final$residuals, order = c(1,0,0))
A$residuals

ARerrors <- A$residuals

acf(ARerrors, lag.max = 10)

# predicting next 12 months errors using arima( order =c(1,0,0))
install.packages("forecast")
library(forecast)
errors_12 <- forecast(A, h = 12)

View(errors_12)

future_errors <- data.frame(errors_12)
class(future_errors)
future_errors <- future_errors$Point.Forecast

# predicted values for new data + future error values 

predicted_new_values <- pred_new + future_errors

write.csv(predicted_new_values, file = "predicted_new_values.csv", row.names = F)
getwd()
