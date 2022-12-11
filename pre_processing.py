import numpy as np
import pandas as pd
import gc

from sklearn.decomposition import PCA
#---------------Utility Functions----------#
def dropColumns(df, drop_cols):
    x=df.drop(columns=drop_cols)
    return x
def myPCA(n_components, df, cols):
    pca = PCA(n_components = n_components, random_state = 10)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(principalComponents)
    return principalDf
#------------------------END---------------#
# Load the Dataset
df1=pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
# Removing Outliers(~97.5 %ile)
df1 = df1[df1['TransactionAmt'] <=800]
df1 = df1[df1['TransactionAmt'] >=10]
# Dropping Target col
target = df1.isFraud
df1 = df1.drop('isFraud', axis = 1)

# Joining train and test to perform further processing
df = df1.append(df2)
null_var=(df.isnull().sum())/df.shape[0]
#dropping transaction amoout
df = df.drop(columns = ['TransactionID'])
# dropping cols with more than 85 % null values
null_var=(df.isnull().sum())/df.shape[0]
drop_cols=null_var[null_var>0.85].keys()
x = dropColumns(df, drop_cols)
#Adding dropped col id_30 as it was found to be imp;
x['id_30'] = df['id_30']
# Freeing up unused space 
del df, df1, df2
gc.collect()

# giving meaning to TransactionDT
day = (x['TransactionDT'] /86400) % 7
day = day.astype(int)
hours = (x['TransactionDT'] / 3600)% 24
hours = hours.astype(int)
date = (x['TransactionDT'] /86400)
date = date.astype(int)

# adding hours and days and date to the dataset
x['hours'] = hours
x['days'] = day
x['date'] = date

#Categorising P_email properly
x.P_emaildomain.fillna("NA",inplace = True)
x.loc[x['P_emaildomain'].isin(['gmail.com','gmail']),'P_emaildomain'] = 'Google'
x.loc[x['P_emaildomain'].str.contains('yahoo'),'P_emaildomain'] = 'Yahoo'
x.loc[x['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com','hotmail.es','hotmail.co.uk','hotmail.de','outlook.es','live.com','live.fr','hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
x.loc[x.P_emaildomain.isin(x.P_emaildomain.value_counts()[x.P_emaildomain.value_counts() <= 500].index),'P_emaildomain'] = 'Others'


#Categorising R_email properly
x.R_emaildomain.fillna("NA",inplace = True)
x.loc[x['R_emaildomain'].isin(['gmail.com','gmail']),'R_emaildomain'] = 'Google'
x.loc[x['R_emaildomain'].str.contains('yahoo'),'R_emaildomain'] = 'Yahoo'
x.loc[x['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com','hotmail.es','hotmail.co.uk','hotmail.de','outlook.es','live.com','live.fr','hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
x.loc[x.R_emaildomain.isin(x.R_emaildomain.value_counts()[x.R_emaildomain.value_counts() <= 500].index),'R_emaildomain'] = 'Others'

#Categorising id_30 properly
x['id_30'].fillna('NA',inplace = True)
x.loc[x['id_30'].str.contains('Android',na = False), 'id_30'] = 'Android'
x.loc[x['id_30'].str.contains('Mac OS',na = False), 'id_30'] = 'Mac'
x.loc[x['id_30'].str.contains('Windows', na = False), 'id_30'] = 'Windows'
x.loc[x['id_30'].str.contains('iOS',na = False),'id_30'] = 'iOS'


#Categorising id_31 properly
x.loc[x['id_31'].str.contains('samsung',na = False),'id_31'] = 'Samsung'
x.loc[x['id_31'].str.contains('firefox',na = False),'id_31'] = 'Firefox'
x.loc[x['id_31'].str.contains('chrome', na = False),'id_31'] = 'Chrome'
x.loc[x['id_31'].str.contains('safari', na = False), 'id_31'] = 'Safari'
x.loc[x['id_31'].str.contains('edge', na = False),'id_31'] = 'Edge'
x.loc[x['id_31'].str.contains('ie', na = False),'id_31'] = 'IE'
x.loc[x['id_31'].str.contains('opera', na = False),'id_31'] = 'Opera'
x['id_31'].fillna('NA',inplace = True)
x.loc[x.id_31.isin(x.id_31.value_counts()[x.id_31.value_counts() < 200].index),'id_31'] = 'Others'


# v cols from 45 - 336 are imputed min - 1 and scaling using MinMaxScaler
v_cols = x.columns[45:336]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for col in v_cols:
    x[col] = x[col].fillna((x[col].min() - 1))
x[v_cols] = scaler.fit_transform(x[v_cols])

#dropping other cols with >70% values
drop_cols=null_var[null_var>0.7].keys()
x = dropColumns(x, drop_cols)

#Imputing categorical with NA and numerical cols with min - 1
for col in x.columns:
    if x[col].dtype == 'object':
        x[col].fillna('NA',inplace = True)
    else :
        x[col].fillna(x[col].min()-1, inplace = True)

# Lable Encoding categorical data
from sklearn.preprocessing import LabelEncoder
for col in x.columns:
    if x[col].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(x[col].values))
        x[col] = lbl.transform(list(x[col].values))

# removing the col names with generic numerical values
x = pd.DataFrame(x.values)

#Applying PCA for the V columns
v_cols = x.columns[45:225]
vdf = x[v_cols]
x_pca = myPCA(30, vdf, v_cols)

#dropping V cols and adding the new V cols which are Dimensionally reduced Data
x.drop(v_cols, axis = 1, inplace = True)
x_transformed = pd.concat([x, x_pca], axis = 1, join = "inner")

#Separating the train and test
df_train= x_transformed.iloc[:429525,:]
df_test = x_transformed.iloc[429525:,:]

#generating the csv files for model training
df_train.to_csv("train_mod.csv")
df_test.to_csv("test_mod.csv")
target.isFraud.to_csv("target_mod.csv")