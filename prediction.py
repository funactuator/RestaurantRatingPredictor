import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns

import matplotlib.ticker as mtick
plt.style.use('fivethirtyeight')
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('zomato.csv')
print(data.head())
print(data.shape)
print(data.dtypes) #checking the data types
print(data.isna().sum()) #Checking null values
#You can use pandas profiling to get an over all overview of the dataset
# import pandas_profiling as pf

# pf.ProfileReport(df)

#Deleting Unnnecessary Columns
#Deleting Unnnecessary Columns
df=data.drop(['url','phone'],axis=1) #Dropping the column like "phone" and "url" and saving the new dataset as "df"
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.duplicated().sum()
#Remove the NaN values from the dataset
df.dropna(how='any',inplace=True)
print(df.isnull().sum())
df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
print(df.columns)
df['cost'].unique()
#zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
df['cost'] = df['cost'].apply(lambda x: x.replace(',','')) #Using lambda function to replace ',' from cost
df['cost'] = df['cost'].astype(float)
print(df['cost'].unique())

print('---'*10)

df.dtypes
#Reading uninque values from the Rate column
df['rate'].unique()
df = df.loc[df.rate !='NEW'] #getting rid of "NEW"
df['rate'].unique()
#Removing '/5' from Rates

df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))

plt.figure(figsize=(17,10))
chains=df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")
plt.show()

x=df['book_table'].value_counts()
colors = ['#800080', '#0000A0']

#trace=go.Pie(labels=x.index,values=x,textinfo="value",marker=dict(colors=colors,line=dict(color='#001000', width=2)))
#layout=go.Layout(title="Table booking",width=600,height=600)
#fig=go.Figure(data=[trace],layout=layout)
#py.iplot(fig, filename='pie_chart_subplots')

#Restaurants delivering Online or not
sns.countplot(df['online_order'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Whether Restaurants deliver online or Not')
plt.show()

#How ratings are distributed
plt.figure(figsize=(9,7))

sns.distplot(df['rate'],bins=20)
df['rate'].unique()
df['rate'].min()
df['rate'].max()
df['rate']=df['rate'].astype(float)
((df['rate']>=1) & (df['rate']<2)).sum()
((df['rate']>=2) & (df['rate']<3)).sum()
((df['rate']>=3) & (df['rate']<4)).sum()
(df['rate']>=4).sum()

slices=[((df['rate']>=1) & (df['rate']<2)).sum(),
        ((df['rate']>=2) & (df['rate']<3)).sum(),
        ((df['rate']>=3) & (df['rate']<4)).sum(),
        (df['rate']>=4).sum()
        ]

labels=['1<rate<2','2<rate<3','3<rate<4','>4']
colors = ['#ff3333','#c2c2d6','#6699ff']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their ratings")

fig.set_size_inches(10,10)
plt.show()

#Types of Services

sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.title('Type of Service')

from plotly.offline import iplot
#trace0=go.Box(y=df['cost'],name="accepting online orders",marker = dict(color = 'rgb(113, 10, 100)',))
#data=[trace0]
#layout=go.Layout(title="Box plot of approximate cost",width=800,height=800,yaxis=dict(title="Price"))
#fig=go.Figure(data=data,layout=layout)
#py.iplot(fig)

plt.figure(figsize=(8,8))
sns.distplot(df['cost'])
plt.show()

#re=regular expression (use for splitting words)

import re

df.index=range(df.shape[0])
likes=[]
for i in range(df.shape[0]):
    array_split=re.split(',',df['dish_liked'][i])
    for item in array_split:
        likes.append(item)

df.index=range(df.shape[0])
df.index
print("Count of Most liked dishes in Bangalore")
favourite_food = pd.Series(likes).value_counts()
print(favourite_food.head(30))

#ax = favourite_food.nlargest(n=20, keep='first').plot(kind='bar',figsize=(18,10),title = 'Top 30 Favourite Food counts ')

#for i in ax.patches:
    #ax.annotate(str(i.get_height()), (i.get_x() * 1.005, i.get_height() * 1.005))

#plt.figure(figsize=(15,7))
rest=df['rest_type'].value_counts()[:20]
#sns.barplot(rest,rest.index)
#plt.title("Restaurant types")
#plt.xlabel("count")

print(df.head())

df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] = 0
print(df.online_order.value_counts())
df.online_order = pd.to_numeric(df.online_order)
df.book_table[df.book_table == 'Yes'] = 1
df.book_table[df.book_table == 'No'] = 0
df.book_table = pd.to_numeric(df.book_table)
print(df.book_table.value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.menu_item = le.fit_transform(df.menu_item)
print(df.head())
my_data=df.iloc[:,[2,3,4,5,6,7,9,10,12]]
my_data.to_csv('Zomato_df.csv')
x = df.iloc[:,[2,3,5,6,7,9,10,12]]
print(x.head())
y = df['rate']
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
from sklearn.metrics import r2_score
y_pred=lr_model.predict(x_test)
print(r2_score(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor
RF_Model=RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=.0001)
RF_Model.fit(x_train,y_train)
y_predict=RF_Model.predict(x_test)
print(r2_score(y_test,y_predict))
#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)


from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))
#Use pickle to save our model so that we can use it later

import pickle
# Saving model to disk
pickle.dump(ET_Model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))