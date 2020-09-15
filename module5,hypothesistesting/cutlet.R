cutlet<- read.csv("C:/Users/USER/Downloads/Cutlets.csv")
cutlets<-cutlet[1:35,1:2] # removing NA values
attach(cutlets)

# normality test
shapiro.test(Unit.A)
shapiro.test(Unit.B)

# variance test
var.test(Unit.A,Unit.B)

# 2 sample t test
t.test(Unit.A,Unit.B,alternative = "two.sided",conf.level = 0.95)
