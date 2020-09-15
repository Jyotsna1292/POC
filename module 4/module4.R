x <- read.csv('C:/Users/USER/Downloads/cars.csv')
x
y<-x[,c("MPG")]
z <- 0
for (i in y) {
  if (i > 38){
    z=z+1
  }
  
  
}
print(z)
38/81

t<-0
for(i in y){
  if(i<40){
    t = t+1
  }
}
t
61/81
s<-0
s

for (i in y) {
  if(i>20 & i<50)
    s = s+1
  }
}
s
warning()
69/81

