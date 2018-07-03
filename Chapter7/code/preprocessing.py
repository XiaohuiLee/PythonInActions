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

#%%
data.describe().T["count"]


#%%
# Attribute constraints
# data.columns
# Index(['MEMBER_NO', 'FFP_DATE', 'FIRST_FLIGHT_DATE', 'GENDER', 'FFP_TIER',
#        'WORK_CITY', 'WORK_PROVINCE', 'WORK_COUNTRY', 'AGE', 'LOAD_TIME',
#        'FLIGHT_COUNT', 'BP_SUM', 'EP_SUM_YR_1', 'EP_SUM_YR_2', 'SUM_YR_1',
#        'SUM_YR_2', 'SEG_KM_SUM', 'WEIGHTED_SEG_KM', 'LAST_FLIGHT_DATE',
#        'AVG_FLIGHT_COUNT', 'AVG_BP_SUM', 'BEGIN_TO_FIRST', 'LAST_TO_END',
#        'AVG_INTERVAL', 'MAX_INTERVAL', 'ADD_POINTS_SUM_YR_1',
#        'ADD_POINTS_SUM_YR_2', 'EXCHANGE_COUNT', 'avg_discount',
#        'P1Y_Flight_Count', 'L1Y_Flight_Count', 'P1Y_BP_SUM', 'L1Y_BP_SUM',
#        'EP_SUM', 'ADD_Point_SUM', 'Eli_Add_Point_Sum', 'L1Y_ELi_Add_Points',
#        'Points_Sum', 'L1Y_Points_Sum', 'Ration_L1Y_Flight_Count',
#        'Ration_P1Y_Flight_Count', 'Ration_P1Y_BPS', 'Ration_L1Y_BPS',
#        'Point_NotFlight'],
#       dtype='object')
chosen_cols = ['FFP_DATE',
               'LOAD_TIME','FLIGHT_COUNT','avg_discount','SEG_KM_SUM','LAST_TO_END']
df = data[chosen_cols]
df_ = df.copy()
df_["L"] = pd.to_datetime(df_["LOAD_TIME"]) - pd.to_datetime(df_["FFP_DATE"])
df_ = df_[df_.columns[-5:]]
# print(df_.head())
rename_cols = ["F","C","M","R","L"]
df_.columns = rename_cols


#%%
zdf = (df_ - df_.mean()) / df_.std()
zdf.head()











