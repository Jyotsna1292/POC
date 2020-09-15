# chi squared test/ customerorderform dataset
customerorderform <-read.csv("C:/Users/USER/Downloads/CustomerOrderform.csv", stringsAsFactors=FALSE,header = TRUE)
custorderform <- customerorderform[1:300,1:4]
attach(custorderform)
 
stacked_data<- stack(custorderform)
attach(stacked_data)
table(values,ind)

chisq.test(table(values,ind)) # p-value = 0.2771

# propotional T test/ fantaloons dataset
fantaloons<- read.csv("C:/Users/USER/Downloads/Fantaloons.csv")
attach(fantaloons)

table1 <- table(Weekdays)
table1
table2 <- table(Weekend)
table2
prop.test(x=c(113,167),n=c(400,400), conf.level = 0.95, alternative = "two.sided")




