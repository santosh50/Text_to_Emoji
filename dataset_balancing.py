# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:47:56 2021

@author: Sritanu
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample

df=pd.read_csv(r'S:\SEM 6\NLP\Project\datasets\twitter to emoji\Train.csv',names=['id','text','label'])
df=df[1:]
df['label'].value_counts()
df['label9']=[1 if (l=='9')  else 0 for l in df['label']]
df['label2']=[1 if (l=='2')  else 0 for l in df['label']]
df['label3']=[1 if (l=='3')  else 0 for l in df['label']]
df['label7']=[1 if (l=='7')  else 0 for l in df['label']]
df['label15']=[1 if (l=='15')  else 0 for l in df['label']]
df['majority']=df['majority']=[1 if (l=='9' or l=='2' or l=='3' or l=='7' or l=='15')  else 0 for l in df['label']]
df_minority=df[df.majority==0]
for i in ['label9','label2','label7','label3','label15']:
    df_majority=df[df[i]==1]
    df_majority=resample(df_majority,replace=False,n_samples=2290,random_state=123)
    df_minority=pd.concat([df_majority,df_minority])
  
df=df_minority.drop(['id','label9','label2','label7','label3','label15','majority'],axis='columns')

df.to_csv(r'S:\SEM 6\NLP\Project\datasets\twitter to emoji\Train_balanced.csv')