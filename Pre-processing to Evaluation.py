
#Nama   : Elsa Aritonang
#NIM    : 215150607111005
#Kelas  : PTI PD-B
#TPO3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score, davies_bouldin_score

# PRE PROCESSING #
#deteksi data kosong dan mengisi data yang kosong
dataset= pd.read_csv('psd/DM/unsdg.csv')
datauji= dataset['reading score']
print(datauji)
print(datauji.describe())
print(datauji.isna().sum())
print (datauji.mean())
dataujiisi = datauji.fillna(value = datauji.mean())
dataset['reading score'] = dataujiisi
print(dataujiisi)

datauji2 = dataset['writing score']
datauji2isi = datauji2.fillna(value = datauji2.mean())
dataset['writing score'] = datauji2isi

print('data kosong =', datauji.isna().sum())
print('data kosong terisi =', datauji.isna().sum()-dataujiisi.isna().sum())

#deteksi outlier dan dapatkan data tanpa outlier
print(datauji)
dataujiisi = datauji.fillna(value=datauji.mean())
print(dataujiisi)
print(dataujiisi.describe())

q1, q3 = np.percentile(dataujiisi, [25, 75])
print(q1)
print(q3)
iqr = q3-q1
bb = q1-(1.5*iqr)
ba = q3+(1.5*iqr)
print('batas bawah : ', bb)
print('batas atas : ', ba)
outlier = dataset[(dataujiisi<bb)|(dataujiisi>ba)]
print('data-data outlier =', outlier)
databersih = dataset[(dataujiisi>=bb)&(dataujiisi<=ba)]
print('data bersih dari outlier = ', databersih)
print(dataujiisi.describe())

#normalisasi data dengan minmax
datanorm  = dataset[['writing score', 'reading score']]
print(dataset)
print(datanorm)
scaler = MinMaxScaler()
scaler.fit(datanorm)
datatrans1 = scaler.transform(datanorm)
print(datatrans1)

# CLUSTERING #
#hierarki clustering
feature_names = ['math score','reading score','writing score']
X=np.array(dataset[feature_names])

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
hc.fit(X)

plt.scatter(X[:,0], X[:,1],c=hc.labels_,s=50,cmap='rainbow')
plt.title('Hirarki Clusters')
plt.xlabel('reading score')
plt.ylabel('writing score')
plt.show()

#dendogram
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.show()

# EVALUASI #
#evaluasi menggunakan dbi & silhouette
print('k= ', hc.n_clusters,'--> koefisiensilhouette =', silhouette_score(X, hc.labels_))
print('k= ', hc.n_clusters,'--> daviesbouldin =', davies_bouldin_score(X, hc.labels_))