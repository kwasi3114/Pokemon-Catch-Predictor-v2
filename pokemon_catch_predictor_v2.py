#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#data collected from Pokemon Emerald
#locations tested: route 101, route 110, route 119, route 120
#sky pillar, victory road
#only pokeballs were used so to not skew catch rate

df = pd.read_csv('pokemon_catch_stats_v2_1 - pokemon_catch_stats_v2.csv')
#df = pd.read_csv('pokemon_catch_stats_v2.csv')

df['Captured'] = df['Captured?'] == 'Yes'
print(df)

#status is noted in the csv file but is not utilized in the prediction
X = df[['Catch Rate', 'Level', 'Health Percent']].values
y = df['Captured']

model = LogisticRegression()
model.fit(X,y)
#print(model.coef_, model.intercept_)

print('Input the catch rate of the Pokemon:')
catch_rate = int(input())

print('Next, enter the level of the Pokemon')
level = int(input())

print('Finally, estimate the health percentage of the Pokemon')
health = int(input())

if(catch_rate < 0 or catch_rate > 255 or level < 1 
   or level > 100 or health < 1 or health > 100):
    print("Invalid entry for one or more of the fields.")
else:
    poke_predictor = model.predict([[catch_rate,level,health]])
    print(poke_predictor)


# In[ ]:





# In[ ]:





# In[ ]:




