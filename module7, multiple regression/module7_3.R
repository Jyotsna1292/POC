toyota <- read.csv("C:/Users/USER/Downloads/ToyotaCorolla.csv")

install.packages("dplyr")
library(dplyr)

corolla <-  select(toyota,Price,Age_08_04,KM,HP,cc,Doors,Gears,Quarterly_Tax,Weight)
attach(corolla)
summary(corolla)

pairs(corolla)
cor(corolla)

#forming model taking all variables
model<-lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight, data = corolla)
summary(model)
# forming model taking individual variables
model_1<-lm(Price~Age_08_04, data = corolla)
summary(model_1)
model_2<-lm(Price~KM, data = corolla)
summary(model_2)

install.packages("GGally")
library(GGally)
ggpairs(corolla)

#for partial correlation matrix
install.packages("corpcor")
library(corpcor)
cor2pcor(cor(corolla))

#diagnostic plots
install.packages("car")
library(car)
plot(model)

#deletion diagonastic for identifying influential variable
influence.measures(model)
influenceIndexPlot(model)
influencePlot(model)

write.csv(corolla, file = "toyota_cor.csv")
getwd()

corolladata<-read.csv(file.choose())
model.1<-lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight,data=corolladata)
summary(model.1) 
summary(corolla)
gc()

#regression after deleting 81st observation
model.2<-lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight,data=corolla[-81,])
summary(model.2) #model has improved to some extent

#variance inflation factor
vif(model) #VIF > 10 = collinearity
vifage<-lm(Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight, data = corolla)
vifkm<-lm(KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight,data=corolla)
vifHP<-lm(HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight, data = corolla)
vifcc<-lm(cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight,data=corolla)
vifDoors<-lm(Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight,data=corolla)
vifGears<-lm(Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight,data=corolla)
vifqt<-lm(Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight,data=corolla)
vifWeight<-lm(Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax,data=corolla)
summary(vifage)
summary(vifkm)
summary(vifHP)
summary(vifcc)
summary(vifDoors)
summary(vifGears)
summary(vifqt)
summary(vifWeight)
#added variable plot, avplot
avPlots(model)
# from plot it is clear that Doors column has no contribution in the model so we can remove it

install.packages("MASS")
library(MASS)
stepAIC(model)
# stepAIC suggesting to built model without Doors and cc column
model.3<-lm(Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight, data = corolla)
summary(model.3)
model.4<-lm(Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight, data =corolla[-81,])
summary(model.4)
#there is no change in model after deleting 81st record so we can consider model.3 as final model
avPlots(model.3)

#trainig and testing model
n=nrow(corolla)
n1=n*0.7
n1
train=sample(1:n,n1)
train
test=corolla[-train,]
test
pred=predict(model.3,newdata = test)
pred
actual=test$Price
actual
error = actual-pred
error
test.rmse=sqrt(mean(error**2))
test.rmse
train.rmse=sqrt(mean(model.3$residuals^2))
train.rmse




