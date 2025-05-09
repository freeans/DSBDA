import pandas as pd
import numpy as np
df=pd.read_csv("/home/mangal/Downloads/heart.csv") # give here your own path for dataset
df
df.shape
df.columns
df.dtypes
df.info()
df.describe()
df.isnull().sum()
df.duplicated().any().sum()
df.drop_duplicates(inplace=True)
df.shape
df.duplicated().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,6)) # try (15 , 10)
sns.heatmap(df.corr(),annot=True)
#sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt= ".2f",cmap="YlGnBu");
plt.title('Degree of Correlation of variables in the dataset')


sns.countplot(x='target',data=df)
plt.xticks([0,1],['less chance','more chance'])
plt.title('Chances of heart disease')
plt.figure(figsize=(15,6))
# we can also get these numbers using the pandas value_counts() method
df['target'].value_counts(normalize=True)
# #### Discrete variable : use seaborn.countplot() or histplot()
sns.countplot(x='sex',data=df )
plt.title('Number of males and females')
plt.xticks([0,1] , ['females','males'])
plt.show()
# #### Discrete variable : use seaborn.countplot() or histplot()
sns.countplot(x='sex',data=df, hue ='target' )
plt.title('chances of heart disease by gender')
plt.xticks([0,1] ,['females','males'])
plt.legend(labels=['less chance','high chance'])
# Note the hue attribute acts as second dimension for the plot
sns.histplot(df['cp'])
# same result can be achieved as: sns.displot ( df['cp'], kind = 'hist')
plt.xticks([0,1,2,3],["typical angina","atypical angina",
"non-anginal pain","asymptomatic"])
plt.xticks(rotation=70)
plt.figure(figsize=(10,7))
plt.show()
list1 = list(df['cp'].value_counts(normalize=True))
list1
plt.pie(list1,labels=["typical angina","non-anginal",
"atypical angina","asymptomatic"],startangle=180, shadow=True,
autopct='%1.1f%%')
plt.show()
# ### Distribution of the chest pain distribution as per the target variable
sns.countplot(x='cp',hue='target',data=df)
# formatting the plot
plt.title('Relation between types of chest pain and number of people having high or low chances of heart attack')
plt.xticks([0,1,2,3],["typical angina","atypical angina","non-anginal pain","asymptomatic"])
plt.legend(labels=['low chance','high chance'])
sns.countplot(x='fbs' , hue='target',data=df)
plt.legend(labels=['low chance','high chance'])
df['fbs'].value_counts(normalize=True)
g= sns.FacetGrid(df,hue="sex",aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.xlabel("resting blood presuure")
plt.legend(labels=["female","male"])
chol= sns.FacetGrid(df,hue="sex",aspect=4)
chol.map(sns.kdeplot,'chol',shade=True)
plt.xlabel("serum cholestrol")
plt.legend(labels=["female","male"])
cat_val=[] # list for storing categorical variables
cont_val=[] # list for storing continuous variables
for column in df.columns:
if df[column].nunique() <= 5:
cat_val.append(column)
else:
cont_val.append(column)
cont_val

cat_val = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df.hist(cont_val, figsize=(17,6))
plt.show()
plt.tight_layout()
df.hist(cat_val, layout=(4, 4), figsize=(20, 20), color="DarkCyan",
grid=True)
plt.show()
sns.pairplot(df , hue = 'target')
