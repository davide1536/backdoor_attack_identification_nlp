
from re import X
from read_dataset import read_dataset
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pylab as pl
import seaborn as sns

'''
Perform clustering on dataset's embedding
'''

y, backdoor_labels, embeddings = read_dataset()
embeddings_del = np.delete(embeddings, [range(15000)], 0)
backdoor_labels_del = np.delete(backdoor_labels, [range(15000)], 0)
# embeddings = np.loadtxt('embedding_array.txt')

print(embeddings_del.shape)
print(backdoor_labels_del.shape)
print(backdoor_labels_del[backdoor_labels_del.size-1])
print(backdoor_labels.shape)
print(embeddings.shape)


#tsne = TSNE()
pca = PCA(n_components=2)
X_embedded = pca.fit_transform(embeddings)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=backdoor_labels, legend='full')
pl.show()
# pl.scatter(X_embedded[:,0], X_embedded[:,1])
# pl.legend(backdoor_labels_del)
# pl.show() 
dbs = DBSCAN()
dbs.fit(embeddings)

labels = dbs.labels_
print(metrics.adjusted_rand_score(backdoor_labels, labels))
print(metrics.adjusted_mutual_info_score(backdoor_labels, labels))
print(metrics.homogeneity_score(backdoor_labels, labels))

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=dbs.labels_, legend='full')
pl.show()


