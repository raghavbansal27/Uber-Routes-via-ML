"""Import required modules and load data file"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

uber = pd.read_csv('Travel_Times.csv')
print(uber.shape)

"""Create train-test split"""
X = uber[['Mean Travel Time (Seconds)']]
y = uber[['Origin Display Name', 'Destination Display Name']]
X_train , X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""Creating classifier object"""
knn = KNeighborsClassifier(n_neighbors=5)

"""Training Classifier"""
knn.fit(X_train, y_train)

"""Testing"""
route_prediction = knn.predict([['2000']])  # entering 2000 seconds
for i in route_prediction:
    for j in i:
        print("Predicted route : ", j)
