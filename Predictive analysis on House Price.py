import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns

data = pd.read_csv("housing.csv")

#1:  finding null values
#data.info()

#2:  Drop null values and saves inplace variable
data.dropna(inplace=True)

#3:  Spliting data into Training and Testing Variables
#-> 3(a):  Split into X, Y seperating features and Target value
X = data.drop(['median_house_value'], axis=1)
Y = data['median_house_value']
#-> 3(b):  Split X & Y into training & Testing variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#4:  Combine both feature and target value training data for analysis
train_data = X_train.join(Y_train)

#5:  Training data Exploration
#-> 5(a): Visualizing features Histograms
train_data.hist(figsize=(12, 8))

#-> 5(b):  HeatMap Generation to see correlation between features and Target Value
plt.figure(figsize=(12,8))
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")


#6:  preprocessing
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1 )
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1 )
train_data['population'] = np.log(train_data['population'] + 1 )
train_data['households'] = np.log(train_data['households'] + 1 )

train_data.hist(figsize=(12, 8))

#visualizing the coordinates
plt.figure(figsize=(12,8))
sns.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='coolwarm')

#7:  Feature Engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

plt.show()

#8:  model training

scaler = StandardScaler()

X_train, Y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']

X_train_s = scaler.fit_transform(X_train)

#reg = LinearRegression()

#reg.fit(X_train_s, Y_train)

#Testing
test_data = X_test.join(Y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1 )
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1 )
test_data['population'] = np.log(test_data['population'] + 1 )
test_data['households'] = np.log(test_data['households'] + 1 )

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

X_test, Y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

X_test_s = scaler.transform(X_test)

#reg.score(X_test_s, Y_test)

#Random Forest
forest = RandomForestRegressor()
forest.fit(X_train_s, Y_train)

forest.score(X_test_s, Y_test)

#total 10 folds 9 for training and 1 for testing

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [30, 50, 100], 
    "max_features": [8, 12, 20],  
    "min_samples_split": [2, 4, 6, 8]
}


grid_search = GridSearchCV(forest, param_grid, cv=5, scoring = "neg_mean_squared_error", return_train_score = True)

grid_search.fit(X_train_s, Y_train)

best_forest = grid_search.best_estimator_

print(best_forest.score(X_test_s, Y_test))

