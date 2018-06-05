#importing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.api as sm
import numpy as np
style.use('fivethirtyeight')

#dataimport
df = pd.read_csv('finalnewcleanedblanks.csv')

#checking 5 first rows
print(df.head())

#delete NULL
df.dropna(inplace=True)

#separating gender, race, and residential_status to make them unique parameters
#gender is Xg
Xg = df.iloc[:,7].values
Xg = np.c_[Xg, np.ones(len(Xg))]
#race is Xr
Xr = df.iloc[:,8].values
Xr = np.c_[Xr, np.ones(len(Xr))]
#residential_status and the rest of data is Xt
Xt = df.iloc[:,9:].values
Y = df.iloc[:,6].values

#test print
print ('This is Xg plus one column of ones')
print (Xg)
print ('This is Xr plus one column of ones')
print (Xr)

#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
Xt[:,0] = labelencoder_X.fit_transform(Xt[:,0])
print ('This is Xt after labeltransform')
print (Xt)
onehotencode = OneHotEncoder(categorical_features = [0])
Xt = onehotencode.fit_transform(Xt).toarray()
print ('This is Xt after onehotencode')
print (Xt)

Xg[:,0] = labelencoder_X.fit_transform(Xg[:,0])
print ('This is Xg after labeltransform')
print (Xg)
onehotencode = OneHotEncoder(categorical_features = [0])
Xg = onehotencode.fit_transform(Xg).toarray()
print ('This is Xg after onehotencode')
print (Xg)

Xr[:,0] = labelencoder_X.fit_transform(Xr[:,0])
print ('This is Xr after labeltransform')
print (Xr)
onehotencode = OneHotEncoder(categorical_features = [0])
Xr = onehotencode.fit_transform(Xr).toarray()
print ('This is Xr after onehotencode')
print (Xr[:20,:])

#datacombination
Xt = np.c_[Xg[:,:-1],Xr[:,:-1],Xt]
print ('New Xt')
print (Xt)
print ('One row of Xt')
print (Xt[0,:])

#calculation
X_1=sm.add_constant(Xt)
results=sm.OLS(Y,X_1).fit()
print (results.params)
print(results.summary())
