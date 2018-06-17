#%%
import pandas as pd
b = [i for i in range(0,5,1)]
a = [i for i in range(6,11,1)]
s = [i for i in range(1,20,1)]
c = range(1,3,1)
d = range(6,10,1)
list(c)+list(d)
# s[list(c) + list(d)]
# s[]
s = pd.Series(s)
s.iloc[list(c)+list(d)]


#%%
import os
outputfile = os.getcwd() + "Chapter6\\output"
outputfile