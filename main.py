
import numpy as np
import methods as m
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train = pd.read_csv("train2.csv", header=None)
test = pd.read_csv("test2.csv", header=None)
y = np.asarray(train[14])
X = np.asarray(train.drop(14, axis=1))
tmp = [1, 3, 5, 6, 7, 8, 9, 13]
imp = SimpleImputer(missing_values=' ?', strategy='most_frequent')
X = imp.fit_transform(X)
test = imp.transform(test)
encoder = LabelEncoder()
for i in tmp:
    X[:, i] = encoder.fit_transform(X[:,[i]])
    test[:, i] = encoder.transform(test[:, [i]])

std = StandardScaler()
std.fit(X)
X = std.transform(X)
min_max = MinMaxScaler()
min_max.fit(X)
X = min_max.transform(X)
GDB = m.GradientBoosting(n_estimators=50)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
GDB.fit(Xtrain, ytrain)
y_pred = GDB.predict(Xtest)
print(accuracy_score(ytest, y_pred))

clf_xg = XGBClassifier(n_estimators=50)
clf_xg.fit(Xtrain, ytrain)
y_pred_xgb = clf_xg.predict(Xtest)
print("XGBoots")
print(accuracy_score(ytest, y_pred_xgb))