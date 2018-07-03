#%%
import pandas as pd
import numpy as np


#%%
inputfile = "Chapter7\\data\\air_data.csv"
data = pd.read_csv(inputfile, encoding="utf-8-sig")
describe_df = data.describe().T
choosed_cols = ["count", "min", "max"]
describe_ = describe_df.copy()
describe_ = describe_[choosed_cols]
describe_["notNUllCount"] = len(data) - describe_["count"]
explore = describe_.sort_values("notNUllCount", ascending = False)


