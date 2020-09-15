
# importing libraries
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

# importing dataset
airlines = pd.read_excel("C:/Users/USER/Downloads/Airlines.xlsx")
Airlines = airlines.iloc[:,1:]

# normalizing data to bring all the variables at same scale
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm = norm_func(Airlines.iloc[:,:])
df_norm.head(5)

# plotting scree plot
k = list(range(2,26))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i).fit(df_norm)

    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
#scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("TWSS");plt.xticks(k)

# by visualizing the scree plot we can see the elbow at 5 and 10, but will take clusters as 10 beacuse of the large dataset

# preparing model for 10 clusters
kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42)
y = kmeans.fit_predict(Airlines)

Airlines["clusters"] = y

print(Airlines['clusters'].unique())
count = Airlines['clusters'].value_counts()






