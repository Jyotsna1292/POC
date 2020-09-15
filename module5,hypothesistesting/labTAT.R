labTAT<-read.csv("C:/Users/USER/Downloads/LabTAT.csv")
LabTAT<-labTAT[1:120,1:4] # removing null values
attach(LabTAT)

# normality test
shapiro.test(Laboratory.1)
shapiro.test(Laboratory.2)
shapiro.test(Laboratory.3)
shapiro.test(Laboratory.4)

# variance test
var.test(Laboratory.1,Laboratory.2)
var.test(Laboratory.2,Laboratory.3)
var.test(Laboratory.3,Laboratory.4)
var.test(Laboratory.4,Laboratory.1)

# ANOVA test
stacked_data<- stack(LabTAT)
View(stacked_data)
colnames(stacked_data)
Anova_results<- aov(values~ind, data = stacked_data)
summary(Anova_results)
?aov






