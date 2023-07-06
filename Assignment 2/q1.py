# ### Imports

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import norm
from math import log2
import warnings
warnings.filterwarnings("ignore")

out = open('q1_results.txt', 'w')

# ### Read data

print("==================== Reading Data ====================", file=out)
iris_data = pd.read_csv('./iris.data', delimiter=',')
iris_data = iris_data.to_numpy()
''''
    classes:
        Iris Setosa      = 1
        Iris Versicolour = 2
        Iris Virginica   = 3
'''

print("The data read is: ", file=out)
print(iris_data, file=out)
for i in range(len(iris_data)):
    if iris_data[i][4] == "Iris-setosa":
        iris_data[i][4] = 1

    elif iris_data[i][4] == "Iris-versicolor":
        iris_data[i][4] = 2
    
    else:
        iris_data[i][4] = 3

# ### Helper funtion

def autolabel(ax, reacts):
    """
    Attach a text label above each bar, displaying its height.
    """
    for rect in reacts:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# ### Applying PCA

print("\n\n==================== Applying PCA ====================", file=out)
pca = PCA(n_components=0.95)
x_train = iris_data[:,:4]
y_train = iris_data[:, -1]
pca.fit(x_train) 
x_train = pca.transform(x_train)
var = pca.explained_variance_ratio_[:]  # percentage of variance explained
labels = ['PC' + str(i + 1) for i in range(len(var))]

print("The features after applying PCA is: ", file=out)
print(x_train, file=out)
fig, ax = plt.subplots(figsize=(15, 7))
plot1 = ax.bar(labels, var)

ax.set_title('PCA Plot')
ax.plot(labels, var)
ax.set_title('Proportion of Variance Explained VS Principal Component')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained')
autolabel(ax, plot1)
plt.savefig('variance_ratio_pca.png')
cumsum = [i for i in var]
nc = -1
for i in range(1, len(cumsum)):
    cumsum[i] += cumsum[i - 1]
    if cumsum[i] >= 0.95 and nc == -1:
        nc = i + 1

fig, ax = plt.subplots(figsize=(15, 7))
plot2 = ax.bar(labels, cumsum)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Variance Ratio s Cumulative sum')
ax.set_xlabel('Number principal components')
ax.set_title('Variance Ratio cumulative sum VS number principal components')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)

ax.axvline('PC' + str(nc), c='red')
ax.axhline(0.95, c='green')
ax.text('PC5', 0.95, '0.95', fontsize=15, va='center', ha='center', backgroundcolor='w')
autolabel(ax, plot2)
print('\n\nNumber of Components selected: {}'.format(nc), file=out)
print('Variance captured: {} %'.format(cumsum[nc - 1] * 100), file=out)
plt.savefig('variance_ratio_cumulative sum.png')

# ### K_means Clustering

class K_means:
    def __init__(self, n_clusters, max_iter=100, random_state=42):
        self.error = None
        self.centroids = None
        self.labels = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialise_centroids(self, X):
        np.random.RandomState(self.random_state)
        np.random.seed(4)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initialise_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)
    
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

def entropy(labels, n):
    len_data = len(labels)
    entr = 0.0
    for i in range(n):
        cnt = np.count_nonzero(labels == i)
        entr -= (cnt/len_data)*log2(cnt/len_data)
    
    return entr

def conditional_entropy(Y, C, k):
    c_entr = entropy(Y, 3)
    cnt_c = len(C)
    for i in range(k):
        cnts = [0, 0, 0]
        cnt_i = 0
        for j, l in enumerate(C):
            if l == i:
                cnts[Y[j]] += 1
                cnt_i += 1
        
        tot = cnts[0]+cnts[1]+cnts[2]
        val = -cnt_i/cnt_c
        if cnts[0] != 0:
            c_entr -= val*(cnts[0]/tot)*log2(cnts[0]/tot)
        if cnts[1] != 0:
            c_entr -= val*(cnts[1]/tot)*log2(cnts[1]/tot)
        if cnts[2] != 0:
            c_entr -= val*(cnts[2]/tot)*log2(cnts[2]/tot)
    return c_entr

def normalized_mutual_info_score(Y, C, k):
    H_Y = entropy(Y, 3)
    H_C = entropy(C, k)
    M_I = conditional_entropy(Y, C, k)
    return (2*M_I)/(H_C+H_Y)


# ### K_means training

def k_means_clustering(k, clustering_dataset, true_labels):
    km = K_means(n_clusters=k, max_iter=2000)
    km.fit(clustering_dataset)
    km_labels = km.predict(clustering_dataset)
    nmi = normalized_mutual_info_score(true_labels, km_labels, k)
    return nmi

# ### Applying K-Means

print("\n\n==================== Applying K-Means ====================", file=out)
normalised_mutual_info = []
for i in range(len(y_train)):
    y_train[i] -= 1

max_nmi = 0
max_k = 0

i = 1
for k in range(2, 9):
    nmi = k_means_clustering(k, x_train, true_labels=y_train)
    normalised_mutual_info.append(nmi)
    print(f"NMI for k = {k}: {nmi}", file=out)

    # finding value of k with max value of nmi
    if nmi > max_nmi:
        max_nmi = nmi
        max_k = k

    i += 1

print(f"\n\nValue of k for which the value of nmi is maximum is {max_k} with nmi {max_nmi}", file=out)
labels = [item for item in range(2, 9)]
fig, ax = plt.subplots(figsize=(15, 7))
plot3 = ax.bar(labels, normalised_mutual_info)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.plot(labels, normalised_mutual_info)
ax.set_title('NMI vs K Plot')
ax.set_xlabel('K')
ax.set_ylabel('Normalised Mutual Info.')
autolabel(ax, plot3)
plt.savefig('k_vs_nmi.png')
maxi_k = max(normalised_mutual_info)

out.close()


