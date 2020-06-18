import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine # load_wine() -> I can't load data and target from it ?¿ :(


"""
In this sample we will try to categorize some wines of which we have some features
Unlike classification, in this case we DON'T HAVE LABELS FOR DATA (UNSUPERVISED), we have to 'find' them!
To achieve this, we will use a CENTROID BASED model: K-MEANS. This method perform 4 steps (automatically with sklearn):
    1) Pick k random points as centroids (cluster centers)
    2) Assign each datapoint to nearest centroid (in this case we use euclidean distance, but others coud be used)
    3) Once all points are assigned, calculate each cluster centroid.
    4) Repeat steps 2 and 3 with calculated centroids until none cluster changes (or max. repetitions reaches)
"""

#region LOAD DATA
#data = load_wine() # I can't load data and target from it ?¿ :(
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print('Feature names')
print(data.columns)
print('Data shape')
print(data.shape)
print(data.head())
print('Target')
print(data.target)
#endregion load data
