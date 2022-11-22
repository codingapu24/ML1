# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
#%%
df=pd.read_csv('C:\\Users\\Hp\\Downloads\\archive (2)\\diabetes.csv')
print(df.head())
#%%   DATA PREPROCESSING 
# 1. REMOVE NULL VALUE 
print(df.isna().sum())
print(df.dropna(inplace=True))
print(df.isna().sum())

#%%
#2. DUPLICATE REMOVE
print(df.drop_duplicates(inplace=True))
#%%
#EDA 
print(df.info())
print(df.describe())    
print(df.columns)
#%%
df1=df
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(df1.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = df1.Outcome
#%%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
#%%
knn_classifier = KNeighborsClassifier(n_neighbors = 11)
knn_classifier.fit(X_train, y_train)
y_pred_knn=knn_classifier.predict(X_test) 


#%%
print("Confusion matrix")
cmat=confusion_matrix(y_test,y_pred_knn)
print(cmat)
#%%

ax = sns.heatmap(cmat, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()
#%%
#Accuracy = (TP+TN)/(TP+FP+FN+TN)
print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_pred_knn)))

#%%
#Precision = TP/(TP+FP)	
print('Precision: {:.2%}'.format(precision_score(y_test, y_pred_knn)))
#%%
#recall=TP / (FN + TP)
print('Recall: {:.2%}' .format( recall_score(y_test, y_pred_knn)))
#%%
print('F1 Score: ')
f1_score(y_test, y_pred_knn)
#%%

