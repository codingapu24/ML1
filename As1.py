import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import  mean_squared_error
#%% ...hine Learning/u
df=pd.read_csv('C:\\Users\\Hp\\Downloads\\archive (1)\\uber.csv')

print(df.head())
#%%   DATA PREPROCESSING 
# 1. REMOVE NULL VALUE 
print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())

#%%
#2. DUPLICATE REMOVE
df.drop_duplicates(inplace=True)
#%%
#EDA 
print(df.info())
print(df.describe())    
print(df.columns)
# -*- coding: utf-8 -*-

#%%
#   REFRAMING THE COLUMNS
df = df[(df.pickup_latitude<90) & (df.dropoff_latitude<90) &
        (df.pickup_latitude>-90) & (df.dropoff_latitude>-90) &
       (df.pickup_longitude<180) & (df.dropoff_longitude<180) &
      (df.pickup_longitude>-180) & (df.dropoff_longitude>-180)]

df.pickup_datetime=pd.to_datetime(df.pickup_datetime)

df['year'] = df.pickup_datetime.dt.year
df['month'] = df.pickup_datetime.dt.month
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour

#%%

print(df.info())
#%% DROP UNNESSESARY COLUMNS
df.drop_duplicates(inplace=True)

df.drop(['Unnamed: 0','key'], axis=1, inplace=True)
df.drop(['pickup_datetime','month', 'hour',], axis=1, inplace=True)
#%%
sns.heatmap(df.corr(),annot=True)
#%%
#nf=df.columns

target = 'fare_amount'
features = [i for i in df.columns if i not in [target]]
#%%
#Checking number of unique rows in each feature
nu = df.drop([target], axis=1).nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

for i in range(df.drop([target], axis=1).shape[1]):
    if nu.values[i]<=24:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\nThe Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

#%%
##OUTERLIERS DETECTION 
sns.distplot(df['fare_amount'])
#%%
sns.distplot(df['pickup_latitude'])
#%%
sns.distplot(df['pickup_longitude'])
#%%
sns.distplot(df['dropoff_longitude'])
#%%
sns.distplot(df['dropoff_latitude'])
#%%

#%%
features1 = nf  # outerliers fnding on numerical features
df1=df.copy()
for i in features1:
    
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df[i] <= (Q3+(1.5*IQR))]
    df = df[df[i] >= (Q1-(1.5*IQR))]
    df = df.reset_index(drop=True)
print(df.head())
print('\nBefore removal of outliers, The dataset had {} samples.'.format(df1.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df.shape[0]))
#%%
#corr r2
#creating a correlation matrix

corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
#%%
X = df.drop([target],axis=1)
Y = df[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
#%%
regr = LinearRegression()
regr.fit(Train_X,Train_Y)
print("Linear Regression score R^2 Score ")
regr.score(Test_X, Test_Y)
y_pred_lr = regr.predict(Test_X)
lr_mse = np.sqrt(mean_squared_error(y_pred_lr, Test_Y))
print("RMSE value for Linear regression is:", lr_mse)

#%%
rfr = RandomForestRegressor(n_estimators = 20, random_state = 101)
rfr.fit(Train_X,Train_Y)
print("RandomForestRegressor Regression score R^2 Score ")
rfr.score(Test_X, Test_Y)
y_pred_rfr = rfr.predict(Test_X)
rfr_mse = np.sqrt(mean_squared_error(y_pred_rfr, Test_Y))
print("RMSE Value for Random Forest Regression is:", rfr_mse)
#%%

