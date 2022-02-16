#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[1]:


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


# In[ ]:





# In[3]:


df = pd.read_csv('source.csv',encoding=r'Big5')


# In[4]:


df['發生時段']


# In[5]:


time_zone = []
for r in df['發生時段']:
    if(r == '00~02' or r == '03~05' or r == '04~06' or r == '05~07'):
        time_zone.append('midnight')
    elif(r == '06~08' or r =='08~10' or r == '09~11' or r == '10~12' or r == '11~13'):
        time_zone.append('morning')
    elif(r == '12~14' or r == '14~16' or r == '15~17' or r == '15~18' or r == '16~18' or r == '17~19'):
        time_zone.append('afternoon')
    else:
        time_zone.append('night')
df['time_zone'] = time_zone
df.loc[df['time_zone']=='morning'].to_csv('morning.csv',encoding='utf-8-sig')
df


# In[6]:


list = []
for r in df['發生地點']:
    list.append(r[0:6])

len(list)


# In[7]:


df['locate'] = list
import pandas as pd

temp = pd.Timestamp('2014-06-23')
print(temp.dayofweek, temp.day_name())


# In[8]:


temp=pd.Timestamp(20201125)
print(temp.dayofweek,temp.day_name())


# In[9]:


list = []
for r in df['發生日期']:
    r = str(r)
    y = str(int(r[:3]) + 1911)
    m = r[3:5]
    d = r[5:7]
    r = y + '-' + m + '-' + d
    temp = pd.Timestamp(r)
    strtemp = ''
    if(temp.dayofweek == 0):
        strtemp = '周一'
    elif(temp.dayofweek == 1):
        strtemp = '周二'
    elif(temp.dayofweek == 2):
        strtemp = '周三'
    elif(temp.dayofweek == 3):
        strtemp = '周四'
    elif(temp.dayofweek == 4):
        strtemp = '周五'
    elif(temp.dayofweek == 5):
        strtemp = '周六'
    elif(temp.dayofweek == 6):
        strtemp = '周日'
    list.append(strtemp)


# In[10]:


list


# In[11]:


df


# In[12]:


for r in range(len(df['locate'])):
    list[r] = df['locate'][r] + list[r]


# In[13]:


df['locate'] = list
train, test = train_test_split(df, test_size = 0.8)
train


# In[14]:


BOW_vectorizer = CountVectorizer() 
BOW_vectorizer.fit(train['locate'])
train_data_BOW_features = BOW_vectorizer.transform(train['locate'])
test_data_BOW_features = BOW_vectorizer.transform(test['locate'])


# In[15]:


train_data_BOW_features


# In[16]:


feature_names = BOW_vectorizer.get_feature_names()
feature_names


# In[17]:


import nltk
import jieba
# build analyzers (bag-of-words)
BOW = CountVectorizer(tokenizer=jieba.cut) 

# apply analyzer to training data
BOW.fit(train['locate'])

train_data_BOW_features = BOW.transform(train['time_zone'])

## check dimension
train_data_BOW_features.shape


# In[18]:


train_data_BOW_features.toarray()


# In[19]:


feature_names = BOW.get_feature_names()
train['locate']


# In[20]:


from sklearn.tree import DecisionTreeClassifier

# for a classificaiton problem, you need to provide both training & testing data
X_train = BOW.transform(train['locate'])
y_train = train['time_zone']

X_test = BOW.transform(test['locate'])
y_test = test['time_zone']

## take a look at data dimension is a good habbit  :)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)


# In[21]:


X_train.toarray()


# In[22]:


## build DecisionTree model
DT_model = DecisionTreeClassifier(random_state=0,criterion='gini')

## training!
DT_model = DT_model.fit(X_train, y_train)

## predict!
y_train_pred = DT_model.predict(X_train)
y_test_pred = DT_model.predict(X_test)

## so we get the pred result
y_test_pred[:20]


# In[23]:


## accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
acc_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)

print('accuracy: {}'.format(round(acc_train, 2)))


# In[24]:


test = BOW.transform(['台北市北投區'])
DT_model.predict(test)[0]


# In[44]:


test = BOW.transform(['臺北市中山區松江路25巷1之1~30號'])
DT_model.predict(test)[0]


# In[26]:


import folium
m = folium.Map(
    location=[25.0431, 121.539723],
    zoom_start=13,
    tiles="stamen terrain"
)
m


# In[27]:


m.save("map.html")


# In[28]:


test = BOW.transform(['臺北市文山區萬美里萬寧街1~30號'])
DT_model.predict(test)[0]


# In[29]:


from geopy.geocoders import Nominatim

address='5F-1, No. 25, Chenggong 2nd Rd., Qianzhen Dist., Kaohsiung City 806, Taiwan'
geolocator = Nominatim(user_agent="Your_Name")
location = geolocator.geocode(address)
print(location.address)
print((location.latitude, location.longitude))


# In[ ]:


from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.geocode("台北市中正區北平西路3號", timeout=10)
location = geolocator.geocode("No.3, Beiping W. Rd., Jhongjheng District, Taipei City 100, Taiwan", timeout=10)


# In[ ]:





# In[30]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Your_Name")
for i in data :
    location = geolocation.geocode(i)
    print((location.latitude, location.longitude))


# folium.Circle(
#     radius=10,
#     location=[point.position.lat, point.position.lng],
#     popup=popup,
#     fill_color=color,
#     fill_opacity=1,
#     color="#555",
#     weight=1
# ).add_to(m)

# In[31]:


import json
# 讀取資料
stations = json.load(open("result.json","r",encoding="utf-8"))

# 創建地圖
m = folium.Map(location=[25.0431, 121.539723],zoom_start=13,tiles="cartodbpositron")

# 將資料點加到地圖上
for station in stations:
    folium.Circle(
        radius=50,
        location=[station["lat"], station["lng"]],
        popup="{} \n {}".format(station["sna"],station["tot"]),
        color="#555",
        weight=1,
        fill_color="#FFE082",
        fill_opacity=1,

    ).add_to(m)
m


# In[ ]:





# In[32]:


# 讀取資料
stations = open('source.csv', 'rb').read()
# 創建地圖
m = folium.Map(location=[25.0431, 121.539723],zoom_start=13,tiles="cartodbpositron")

# 將資料點加到地圖上
for station in stations:
    folium.Circle(
        radius=50,
        location=[station["lat"], station["lng"]],
        popup="{} \n {}".format(station["sna"],station["tot"]),
        color="#555",
        weight=1,
        fill_color="#FFE082",
        fill_opacity=1,

    ).add_to(m)
m


# In[ ]:





# In[ ]:





# In[33]:


times=[]
DT_model.predict(test)[0]
for r in df['發生地點']:
    test = BOW.transform([r])  
    times.append(DT_model.predict(test)[0])
times[:10]


# In[34]:


from geopy.geocoders import Nominatim
list = []
for r in df['發生地點']:
    list.append(r[0:6])

geolocation = Nominatim(user_agent="just_test")

arr = []


for i in list[0:10] :
    location = geolocation.geocode(i)
    arr.append([location.latitude,location.longitude,i])
arr = np.array(arr)
arr


# In[ ]:





# In[35]:


print(arr[0,0])


# In[36]:


list


# In[ ]:





# In[ ]:





# In[37]:


import folium
m = folium.Map(
    location=[arr[0,0], arr[0,1]],
    zoom_start=13,
     tiles="Stamen Terrain"
)
m


# In[38]:


m.save("map.html")


# In[39]:



m = folium.Map(location=[arr[0,0], arr[0,1]],zoom_start=8) 


# In[40]:


import folium
taipei = folium.Map(location=[arr[0,0], arr[0,1]],zoom_start=8)

# 添加標記
folium.Marker(
    location=[arr[0,0], arr[0,1]], # 位置
    popup= arr[0,2] + times[0], 
    icon=folium.Icon(icon='cloud') ,
    color="#555",
    weight=10,
    fill_color="#FFE082",
    fill_opacity=1,
).add_to(m)


# In[41]:


m


# In[42]:


from geopy.geocoders import Nominatim
list = []
for r in df['發生地點']:
    list.append(r[0:6])
geolocation = Nominatim(user_agent="just_test")

arr = []


for i in list[0:10] :
    location = geolocation.geocode(i)
    arr.append([location.latitude,location.longitude,i])
arr = np.array(arr)
arr


# In[43]:


print(arr[0,0])


# In[ ]:




