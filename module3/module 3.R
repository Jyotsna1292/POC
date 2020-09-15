
getwd()
setwd("C:/Users/USER/Downloads")
C:\Users\USER\Downloads
x <- read.csv("C:/Users/USER/Downloads/Q1_a.csv")

x
#x <- read.csv(file.choose())
View(x)
install.packages("moments")
attach(x)
library(moments)
skewness(x, na.rm = 'FALSE')
hist(x$speed)
hist(x$dist)
kurtosis(x)

y <- read.csv('C:/Users/USER/Downloads/Q2_b.csv')
y
View(y)
hist(y$SP)
skewness(y)
hist(y$WT)
