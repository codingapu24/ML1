# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#%% ...
df=pd.read_csv('C:\\Users\\Hp\\Downloads\\archive (4)\\spam.csv')
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

#%%
df['Category']=df['Category'].map({'ham': 0, 'spam': 1})
print(df.head())

#%%
X_train, X_test, y_train, y_test= train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=0)
print(X_train.head()) 
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix_train = tfidf.fit_transform(X_train)
tfidf_matrix_valid= tfidf.transform(X_test)
print(tfidf_matrix_train.shape)
#%%
knn_classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p=1)
knn_classifier.fit(tfidf_matrix_train, y_train)
y_pred_knn=knn_classifier.predict(tfidf_matrix_valid)
print('Accuracy Score:', '{:.2%}'.format(accuracy_score(y_test, y_pred_knn)))
cmat_knn=confusion_matrix(y_test, y_pred_knn)
print(cmat_knn)

ax = sns.heatmap(cmat_knn, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
print(ax.set_xlabel('\nPredicted Values'))
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()
#%%

svc = SVC(kernel='rbf', C=10)
svc.fit(tfidf_matrix_train, y_train)
y_pred_svc= svc.predict(tfidf_matrix_valid)
print('Accuracy Score:{:.2%}'.format(accuracy_score(y_test, y_pred_svc)))
cmat_svc=confusion_matrix(y_test, y_pred_svc)
cmat_svc
ax = sns.heatmap(cmat_svc, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
print(ax.set_xlabel('\nPredicted Values'))
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()
