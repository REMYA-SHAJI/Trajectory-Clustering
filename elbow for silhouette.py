'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()
X = iris.data
k = range(1,11)
def eblow(df, n):
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()
eblow(X,k)





elbow


from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

K = range(1,50)
KM = [KMeans(n_clusters=k).fit(dt_trans) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(dt_trans, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/dt_trans.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(dt_trans)**2)/dt_trans.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')

'''

from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X = load_iris().data
y = load_iris().target
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
'''
import random
import numpy as np
def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,10)
    Wks = len(ks)
    Wkbs = len(ks)
    sk = len(ks)
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

X = init_board_gauss(200,3)
ks, logWks, logWkbs, sk = gap_statistic(X)


import pylab as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
 
iris = load_iris()

k = range(1,11)
print iris
kMeansVar  = [KMeans(n_clusters = c,init = 'k-means++').fit(iris.data) for c in k]
centroids = [X.cluster_centers_ for X in kMeansVar]
k_euclid = [cdist(df.values, cent) for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df.values)**2)/df.values.shape[0]
bss = tss - wcss
plt.plot(bss)
plt.show()


centr_lst = [cc.cluster_centers_ for cc in clusters]
 
k_distance = [cdist(iris.data, cent, 'euclidean') for cent in centr_lst]
clust_indx = [np.argmin(kd,axis=1) for kd in k_distance]
distances = [np.min(kd,axis=1) for kd in k_distance]
avg_within = [np.sum(dist)/iris.data.shape[0] for dist in distances]
 
with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
to_sum_square = np.sum(pdist(iris.data) ** 2)/iris.data.shape[0]
bet_sum_square = to_sum_square - with_in_sum_square
 
kidx = 2
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k, avg_within, 'g*-')
ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, \
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering (IRIS Data)')
print "adsSSSSSSSSSSSS"
'''
