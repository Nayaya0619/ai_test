#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


import chardet    
rawdata = open('source.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
print(charenc)


# In[4]:


df = pd.read_csv('source.csv',encoding=r'Big5')


# In[ ]:




