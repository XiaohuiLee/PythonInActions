#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



#%%
k = 5
inputfile = 'Chapter7//data//zscoreddata.xls'
data = pd.read_excel(inputfile)
data.head()

#%%
km = KMeans(n_clusters=k, n_jobs = 3)
km.fit(data)
#%%
se = pd.Series(km.labels_)
counts = se.value_counts().sort_index()


centers = pd.DataFrame(km.cluster_centers_)
centers.columns = data.columns
centers["COUNTS"] = counts
# km.cluster_centers_
centers