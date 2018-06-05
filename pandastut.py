import pandas as pd                 #import pandas to read files
import matplotlib.pyplot as plt     #import matplotlib to generate plots
from matplotlib import style
import statsmodels.api as sm        #import statsmodels for OLS regression
import numpy as np                  #import numpy
style.use('fivethirtyeight')

#data is imported using pandas
df = pd.read_csv('sumperoccurencedemocleaned.csv')

#code below will generate the first 5 rows of imported file
print(df.head())

#code below will delete rows with NULL value, as it will interfere with the calculation process
df.dropna(inplace=True)
 
#the intention is to separate gender, race, and residential_status to make them unique parameters
#during separation, one hot encoding will be applied to create unique values of each value of the string-valued columns (gender, race and residential status)
#later on, these three columns will be joined again together

#creation of Xs for regression, which are independent variables
#Xg corresponds to gender
Xg = df.iloc[:,4].values
#numpy is used to add another column for the purpose of OneHotEncoding processing
Xg = np.c_[Xg, np.ones(len(Xg))]
#Xr corresponds to race
Xr = df.iloc[:,5].values
#numpy is used to add another column for the purpose of OneHotEncoding processing
Xr = np.c_[Xr, np.ones(len(Xr))]
#Xt corresponds to residential_status and the rest of data
Xt = df.iloc[:,6:-1].values

#creation of Y for regression
Y = df.iloc[:,31].values

#printing of sum_amount, which will be Y for regression as a dependent variable
print ('This is sum_amount')
print ('Y')

#printing is done to see Xg has an additional column of ones and encompasses all values
print ('This is Xg plus one column of ones')
print (Xg)
#printing is done to see Xg has an additional column of ones and encompasses all values
print ('This is Xr plus one column of ones')
print (Xr)

#On the following, categorical data such as race, gender and residential status are processed.

#LabelEncoder is used to convert string values into integer.
#If there are 3 types of values in the column (PR, Foreigner, Sporean), integers will be 0,1,2

#OneHotEncoder is used to convert the categorical values into a matrix where each column corresponds to one possible value of one feature.

#LabelEncoder and OneHotEncoder is imported
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
onehotencode = OneHotEncoder(categorical_features = [0])

#Xt
Xt[:,0] = labelencoder_X.fit_transform(Xt[:,0])
#checking if labelencoder is successful
print ('This is Xt after labeltransform')
print (Xt)
Xt = onehotencode.fit_transform(Xt).toarray()
#checking if onehotencoder is successful
print ('This is Xt after onehotencode')
print (Xt)

#Xg
Xg[:,0] = labelencoder_X.fit_transform(Xg[:,0])
#checking if labelencoder is successful
print ('This is Xg after labeltransform')
print (Xg)
Xg = onehotencode.fit_transform(Xg).toarray()
#checking if onehotencoder is successful
print ('This is Xg after onehotencode')
print (Xg)

#Xr
Xr[:,0] = labelencoder_X.fit_transform(Xr[:,0])
#checking if labelencoder is successful
print ('This is Xr after labeltransform')
print (Xr)
Xr = onehotencode.fit_transform(Xr).toarray()
#checking if onehotencoder is successful
print ('This is Xr after onehotencode')
print (Xr[:20,:])

#columns after onehotencoding process are ordered alphabetically
#eg. Chinese, Indian, Malay, Others

#recombining gender, race and residential status into X
X = np.c_[Xg[:,:-1],Xr[:,:-1],Xt]
#prints new X
print ('New X')
print (Xt)
#prints one row data of X
print ('One row of X')
print (Xt[0,:])

#calculation using statsmodels
X_1=sm.add_constant(X)
results=sm.OLS(Y,X_1).fit()
coeff = (results.params[1:])
print (results.params)
print (results.summary())

#creation of axis
para = ('Female', 'Male', 'Chinese', 'Indian', 'Malay', 'Others', 'Foreigner', 'PR', 'Singapore citizen', 'days_in_clinic', 'medical_history_1', 'medical_history_2', 'medical_history_3', 'medical_history_4', 'medical_history_5','medical_history_6', 'medical_history_7', 'preop_medication_1', 'preop_medication_2', 'preop_medication_3', 'preop_medication_4', 'preop_medication_5', 'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2', 'lab_result_3', 'weight', 'height')
y_pos = np.arange(len(para))
 
#creation of bar chart
plt.bar(y_pos, coeff, align='center', alpha=0.5)
plt.xticks(y_pos, para, rotation=90)
plt.ylabel('Degree of effect to bill amount')
plt.title('Effect of Parameters to Bill Amount')

#shows plot
plt.show()
