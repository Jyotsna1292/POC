
# Load the 50_Startups dataset
startup=read.csv("C:/Users/USER/Downloads/50_Startups.csv")

# creating dummy for state column
install.packages("dummies")
library(dummies)

startup <- dummy.data.frame(startup, names = c("State") , sep = ".")
attach(startup)

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
startup_norm <- as.data.frame(lapply(startup, normalize))

# create training and test data
startup_train <- startup_norm[1:35, ]
startup_test <- startup_norm[36:50, ]

## Training a model on the data ----
# train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# simple ANN with only a single hidden neuron
startup_model <- neuralnet(formula = Profit ~ R.D.Spend + Administration +
                              Marketing.Spend + State.California + State.Florida + 
                              State.New.York ,data = startup_train)
                           


# visualize the network topology
plot(startup_model)

## Evaluating model performance 

----
  # obtain model results
  results_model= NULL  
results_model <- compute(startup_model, startup_test[1:6])

# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, startup_test$Profit) # 0.776

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
startup_model2 <- neuralnet(formula = Profit ~ R.D.Spend + Administration +
                              Marketing.Spend + State.California + State.Florida + 
                              State.New.York ,data = startup_train, hidden = 5)


# plot the network
plot(startup_model2)

# evaluate the results as we did before
model_results2 <- compute(startup_model2, startup_test[1:6])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, startup_test$Profit) # 0.846















