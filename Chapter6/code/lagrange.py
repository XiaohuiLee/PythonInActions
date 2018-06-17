#%%
import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
inputData = "Chapter6/data/missing_data.xls"
data = pd.read_excel(inputData,header = None)
print(data.head())
print(data.columns)
print(data[1])
#%%
# 拉格朗日插值
def interpolate(series, index, k = 5):
    chosen_index = list(range(index - k, index)) + list(range(index + 1, index+k+1))
    se = series[chosen_index]
    se = se[se.notnull()]
    print(se)
    return lagrange(se.index, list(se))(index)

#%%
for i in data.columns:
    series = data[i]
    for j in range(len(series)):
        if np.isnan(series[j]):
            series[j] = interpolate(series, j)
#%%
print(data)