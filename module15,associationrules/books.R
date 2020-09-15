
install.packages("arules")
library(arules)

# importing dataset
books <- read.csv("C:/Users/USER/Downloads/book.csv")


install.packages("arueslViz")
library("arulesViz") # for visualizing rules

# making rules using apriori algorithm 
# Keep changing support and confidence values to obtain different rules

# Building rules using apriori algorithm
arules <- apriori(books, parameter = list(support=0.002,confidence=0.6,minlen=2))
arules # 11242 rules
inspect(head(sort(arules,by="lift"))) # to view we use inspect 

# Viewing rules based on lift value

# Overal quality 
head(quality(arules))

# Different Ways of Visualizing Rules
plot(arules)

plot(arules,method="grouped")
plot(arules[1:20],method = "graph") # for good visualization try plotting only few rules

###############################################################################
# changing the support and confidence value and obtaining rules

# Building rules using apriori algorithm
arules <- apriori(books, parameter = list(support=0.02,confidence=0.8,minlen=3))
arules # 11132 rules
inspect(head(sort(arules,by="lift"))) # to view we use inspect 

# Viewing rules based on lift value

# Overal quality 
head(quality(arules))

# Different Ways of Visualizing Rules
plot(arules)

plot(arules,method="grouped")
plot(arules[1:20],method = "graph") # for good visual



