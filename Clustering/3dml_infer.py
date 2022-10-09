import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


data_folder = "/home/goksan/Downloads/3D_ML/"
model_name = "urban_classifier.gox"
val_dataset = "3DML_validation.xyz"


val_pcd = pd.read_csv(data_folder+val_dataset, delimiter=' ')
val_pcd.dropna(inplace=True)
val_labels = val_pcd['Classification']
feature_names = ['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2',
                 'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']
val_features = val_pcd[feature_names]


val_features_train, val_features_test, val_labels_train, val_labels_test = train_test_split(
    val_features, val_labels, test_size=0.9)

loaded_model = pickle.load(open(data_folder + model_name, 'rb'))
predictions = loaded_model.predict(val_features_test)
print(classification_report(val_labels_test, loaded_predictions,
      target_names=['ground', 'vegetation', 'buildings']))
