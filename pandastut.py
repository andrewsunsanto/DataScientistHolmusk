#importing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.api as sm
import numpy as np
style.use('fivethirtyeight')

#dataimport
df = pd.read_csv('sumperoccurencedemocleaned.csv')

#checking 5 first rows
print(df.head())

#delete NULL
df.dropna(inplace=True)

#separating gender, race, and residential_status to make them unique parameters
#gender is Xg
Xg = df.iloc[:,4].values
Xg = np.c_[Xg, np.ones(len(Xg))]
#race is Xr
Xr = df.iloc[:,5].values
Xr = np.c_[Xr, np.ones(len(Xr))]
#residential_status and the rest of data is Xt
Xt = df.iloc[:,6:-1].values
Y = df.iloc[:,31].values
print ('This is sum_amount')
print ('Y')

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
coeff = (results.params[1:])
print (results.params)
print (results.summary())

para = ('Female', 'Male', 'Chinese', 'Indian', 'Malay', 'Others', 'Foreigner', 'PR', 'Singapore citizen', 'days_in_clinic', 'medical_history_1', 'medical_history_2', 'medical_history_3', 'medical_history_4', 'medical_history_5','medical_history_6', 'medical_history_7', 'preop_medication_1', 'preop_medication_2', 'preop_medication_3', 'preop_medication_4', 'preop_medication_5', 'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2', 'lab_result_3', 'weight', 'height')
y_pos = np.arange(len(para))
 
plt.bar(y_pos, coeff, align='center', alpha=0.5)
plt.xticks(y_pos, para, rotation=90)
plt.ylabel('Degree of effect to bill amount')
plt.title('Effect of Parameters to Bill Amount')
 
plt.show()
