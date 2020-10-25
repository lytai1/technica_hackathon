import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/titanic.csv")
print(df)
df.drop(['Name'], axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female': 1})
# df = df[['Survived', 'Sex']]
print(df)

features = list(df.columns[1:])
labels = ['Survived', 'Not Survived']
data = df[df.columns[1:]].values.tolist()
target = list(df['Survived'].map({True:1, False:0}))
print(len(features), "Features: ", features)
print(len(data), 'Data: ', data)
print(len(target), 'Target: ', target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

accuracy = sum([(1 if y_pred[i]>0.5 else 0) == y_test[i] for i in range(len(y_pred))])/len(y_pred)
print("Accuracy of linear regression model = ", accuracy)