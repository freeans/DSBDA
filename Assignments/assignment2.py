import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('AirQuality.csv',sep = ';')
# check the columns names and the respective datatypes
df.info()
df.describe()
# Examine some sample rows ; head shows top five rows by default
df.head()
# are having numeric values; these values contain commas ',' instead of '.'
df.shape # check total number of rows and columns

# Ckeck for missing values in each column ; by finding counts
df.isnull().sum()

# lets find the count using following command
df['Unnamed: 15'].isnull().sum()
df['Unnamed: 16'].isnull().sum()
# This count shows that all values in these two columns are null ;
# hence lets drop these two columns
df.drop(['Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)
df.shape
df.isnull().sum()
# We will print all the rows having any of its value as NaN
df[df.isnull().any(axis=1)]
# The above output shows that these rows of missing values are having NaN in all the columns
df.dropna(how = 'all', inplace = True)
# Following is Syntax of dropna()
# we will check if the rows sre droped by printing these rows
df[df.isnull().any(axis=1)]
column_list = df.columns.values.tolist()
print(column_list)
# Lets check unique values in each columns
# column_list = df.columns.values.tolist()
for column_name in column_list:
print ("\n", column_name)
print ( df[column_name].value_counts(dropna = False ) ) # print count ALL unique values in each column
# dropna argument is optional, shows the count of NA
## ERROR CORRECTION
# in our dataset the column 'CO(GT)' ,'C6H6(GT)', 'T' , 'RH' , 'AH' contain numeric values but contain a comma in each value; Also it is observed that the first column is not showing its datatype as date.
df.info()
print(df['CO(GT)'])
print( df['C6H6(GT)'])
print(df['RH'])
print(df['AH'])
print(df['T'])
# Note all above column we need to replace ',' with '.' and convert them to numeric from string
# ERROR CORRECTION column formatting: 'C6H6(GT)'
j = 'CO(GT) C6H6(GT) T RH AH'.split()
print(j)
df.replace(to_replace=',',value='.',regex=True,inplace=True)
for i in j :
df[i] = pd.to_numeric(df[i],errors='coerce')
df.info()
# following output shows the datatypes of corrected columns changed now
df.head()
scaler = StandardScaler()
Numerical_col = df.select_dtypes(exclude = [np.object_ , np.datetime64 ] )
for col in Numerical_col:
df[[col]] = scaler.fit_transform(df[[col]])
df.head()
df.info()
# DATA TRANSFORMATION :
# date is stored as string. We will change string type to date
# Formatting Date and Time to datetime type
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Time'] = pd.to_datetime(df['Time'],format= '%H.%M.%S' ).dt.time
# Series.dt.time: Returns numpy array of datetime.time objects.
df.head()
df.info()
df.info() 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
heart_data = pd.read_csv('/home/mangal/Downloads/heart.csv') # put her path of your dataset
# print first 5 rows of the dataset
heart_data.head()
# print last 5 rows of the dataset
heart_data.tail()
# number of rows and columns in the dataset
heart_data.shape
# info about the data
heart_data.info()
# checking for missing values
heart_data.isnull().sum()
# statistical measures about the data
heart_data.describe()
# checking the distribution of "target" variable
heart_data['target'].value_counts()
# 1 --> Defective Heart
# 0 --> Healthy Heart
# Splitting the Features and Target
X= heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)
print(Y)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(X_test.head())
# Logistic Regression
model = LogisticRegression(max_iter=1050)
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
# Model Evaluation: finding accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracyy on training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
type(X_test)
print(X_test_prediction)

input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3) # single instance of 13 features

input_data_as_numpy_array= np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
test_df = pd.DataFrame(input_data_reshaped, columns = X_test.columns )
prediction = model.predict(test_df)
print(prediction)
if (prediction[0]== 0):
print('The Person does not have a Heart Disease')
else:
print('The Person has Heart Disease')
