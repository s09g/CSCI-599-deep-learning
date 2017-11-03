# -*- coding: utf-8 -*-
import numpy as np
from sys import argv
import os

_, K, D, N, path, *_ = argv

def load_dataset(path):
    import pickle
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'][:1000], np.array(d[b'labels'][:1000])

def split_dataset(data, labels, N):
    X_test, X_train = data[:N], data[N:]
    y_test, y_train = labels[:N], labels[N:]
    return X_test, X_train, y_test, y_train

def rgb2gray(rgb):
    grayscale = np.row_stack((np.eye(1024) * 0.299, 
                          np.eye(1024) * 0.587,
                          np.eye(1024) * 0.114))
    return np.dot(rgb, grayscale)

def pca(X_train, X_test, D):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=D, svd_solver="full")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

def knn_predict(X_test, X_train, y_train, K, N):
    def K_largest(arr):
        return np.argpartition(arr, -K)[-K:]
    def classify(arr):
        count = {}
        ans, curt = 0, 0
        for i in range(K):
            n = y_train[arr[1:]][i]
            weight = weights[arr[0]][arr[1:]][i]
            count[n] = count[n] + weight if n in count else weight
            if count[n] > curt:
                curt, ans = count[n], n
        return ans
    
    from scipy.spatial.distance import cdist
    weights = 1 / cdist(X_test, X_train)
    knn = np.apply_along_axis(K_largest, 1, weights)
    knn = np.column_stack((range(N), knn))
    return np.apply_along_axis(classify, 1, knn)

def output():
    with open("7637466802.txt", 'w') as f:
        for i in range(N):
            f.write(str(y_pred[i])+" "+str(y_test[i])+"\n")

K, D, N = int(K), int(D), int(N)           
data, labels = load_dataset(path)
X_test, X_train, y_test, y_train = split_dataset(data, labels, N)
X_train, X_test = rgb2gray(X_train), rgb2gray(X_test)
X_train, X_test = pca(X_train, X_test, D)
y_pred = knn_predict(X_test, X_train, y_train, K, N)
output()