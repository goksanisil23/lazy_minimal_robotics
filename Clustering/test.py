from turtle import pd
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


a = np.array([1, 2, 3, 4, 5, 6])
a = a.reshape(-1, 1)
labels = np.array([0, 0, 1, 1, 2, 2])

a_scaled = MinMaxScaler().fit_transform(a)
print(a_scaled)


a_train, a_test, labels_train, labels_test = train_test_split(
    a_scaled, labels, test_size=0.4)

print(a_train)
print(a_test)
print(labels_train)
print(labels_test)
