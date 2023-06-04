import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from IPython.display import display

df = pd.read_csv('USArrests.csv')
# df.head()
# df.sample(5)

col = ['Murder','Assault', 'UrbanPop', 'Rape']

scatter_matrix(df[col], alpha=0.05, figsize=(6, 6)) #scatter matrix and histograms

print(" matrix and histograms")
print(df[col].corr()
)
print(" _____")
plt.show()


dataNorm = preprocessing.MinMaxScaler().fit_transform(df[col].values)

# Calculate the distances between each data set,
# i.e. rows of the data_for_clust array
# Euclidean distance is calculated (default)
data_dist = pdist(dataNorm, 'euclidean')
# Main function of hierarchical clustering
# Combining elements into clusters and saving to
# special variable (used below to render
# and allocate the number of clusters
data_linkage = linkage(data_dist, method='average')


# Elbow method. Allows you to estimate the optimal number of segments.
# Shows sum within group variances
last = data_linkage[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2 
print("Recommended number of clusters:", k)



#функция построения дендрограмм
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

nClust=6

#build a dendrogram
fancy_dendrogram(
    data_linkage,
    truncate_mode='lastp',
    p=nClust, 
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
)
plt.show()


# hierarchical clustering
clusters=fcluster(data_linkage, nClust, criterion='maxclust')
x=0
y=1
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=clusters, cmap='flag')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.show()


df['I'] = clusters
res = df.groupby('I')[col].mean()
res['Количество'] = df.groupby('I').size().values

# We display the results in the form of a table
print(res)

# Displaying data for a specific cluster (for example, cluster 1)

print(df[df['I'] == 4])


# build clustering using KMeans method
km = KMeans(n_clusters=nClust).fit(dataNorm)

# output the resulting distribution over clusters
# also the number of the cluster to which the line belongs, since the numbering starts from zero, we output adding 1
km.labels_ +1

x=0
y=1
centroids = km.cluster_centers_
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=km.labels_, cmap='flag')
plt.scatter(centroids[:, x], centroids[:, y], marker='*', s=300,
            c='r', label='centroid')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.show()


# add cluster numbers to the original data
df['KMeans']=km.labels_+1
res=df.groupby('KMeans')[col].mean()
res['Количество']=df.groupby('KMeans').size().values
print(res)


print(df[df['KMeans'] == 5])

