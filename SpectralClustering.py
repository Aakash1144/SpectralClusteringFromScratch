# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:49:26 2021

@author: aakasharora
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from scipy.sparse import csr_matrix

df_org = pd.read_csv('http://cs.joensuu.fi/sipu/datasets/jain.txt', delimiter = "\t",header = None)

df_org.head()
df=df_org.iloc[:,:2]
inputData=df.to_numpy()


W = pairwise_distances(inputData, metric="euclidean")


vectorizer = np.vectorize(lambda x: 1 if x < 10 else 0)
W = np.vectorize(vectorizer)(W)
print(W)
# convert to sparse matrix (CSR method)
S = csr_matrix(W)
print(S)
# reconstruct dense matrix

# degree matrix
D = np.diag(np.sum(np.array(S.todense()), axis=1))
print('degree matrix:')
print(D)
# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)

e, v = np.linalg.eig(L)
# eigenvalues
print('eigenvalues:')
print(e)
# eigenvectors
print('eigenvectors:')
print(v)

e1=np.where(e == np.partition(e, 1)[1])
# the second smallest eigenvalue
np.partition(e, 1)[1]



H=v[:,[0,1]]


kmeans = KMeans(n_clusters=2)
kmeans.fit(H)
y_pred=kmeans.labels_


y_actual=np.array(df_org.iloc[:,2].tolist())

dot_size=50
plt.scatter(inputData[:, 0], inputData[:, 1],c=y_pred,cmap='viridis', s=dot_size)

plt.show()
from sklearn.metrics import accuracy_score

# Changing the cluster assignment into 1 & 2 (As provided in question) from 0 & 1
y_pred[y_pred == 0] = 1
y_pred[y_pred == 1] = 2

accuracy_score(y_actual,y_pred)



