
# import dataset
mdata <- read.csv("C:/Users/USER/Downloads/mdata.csv")

install.packages("mlogit")
library(mlogit)
install.packages("nnet")
library(nnet)

table(mdata$prog)

mdata1 <- mdata[ ,2:11] # removing unnecessary column

# preparing model
mdata_prog <- multinom(prog~., data = mdata1)
summary(mdata_prog)

#mdata_prog  <- relevel(mdata1$prog, ref= "general")  # change the baseline level

##### Significance of Regression Coefficients###
z <- summary(mdata_prog)$coefficients / summary(mdata_prog)$standard.errors
p_value <- (1-pnorm(abs(z),0,1))*2

summary(Mode.choice)$coefficients
p_value

# predict probabilities
prob <- fitted(mdata_prog)
prob

# Find the accuracy of the model

class(prob)
prob <- data.frame(prob)
View(prob)
prob["pred"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)

prob$pred <- pred_name
View(prob)

# Confusion matrix
table(pred_name,mdata1$prog)


# Accuracy 
mean(pred_name==mdata1$prog) # 63.5 %







