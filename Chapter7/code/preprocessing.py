# data preprocessing
#%%
import pandas as pd
import numpy as np
#%%
inputfile = "Chapter7\\data\\air_data.csv"
data = pd.read_csv(inputfile, encoding="utf-8-sig")

#%%
data = data[data["SUM_YR_1"].notnull() & data["SUM_YR_2"].notnull() ]

mask1 = data["SUM_YR_1"] != 0
mask2 = data["SUM_YR_2"] != 0


# data.describe().T["count"]