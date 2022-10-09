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
dataset = "3DML_urban_point_cloud.xyz"
val_dataset = "3DML_validation.xyz"

pcd = pd.read_csv(data_folder+dataset, delimiter=' ')
val_pcd = pd.read_csv(data_folder+val_dataset, delimiter=' ')

pcd.dropna(inplace=True)
val_pcd.dropna(inplace=True)

# Extract features and labels from the data
labels = pcd['Classification']
val_labels = val_pcd['Classification']

# not so good generalization results
# feature_names = ['X', 'Y', 'Z', 'R', 'G', 'B']
# Gives better generalization results
feature_names = ['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2',
                 'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']
features = pcd[feature_names]
val_features = val_pcd[feature_names]

# Use small portion (10%) of the data from the validation dataset to enrich the training and increase generalizability
val_features_train, val_features_test, val_labels_train, val_labels_test = train_test_split(
    val_features, val_labels, test_size=0.9)

labels = pd.concat([labels, val_labels_train])
features = pd.concat([features, val_features_train])

# Normalize the features
features_scaled = MinMaxScaler().fit_transform(features)

# Determine the train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.4)

# Train
rf_classifier = RandomForestClassifier(n_estimators=10, verbose=1)
knn_classifier = KNeighborsClassifier()
mlp_classifier = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)

rf_classifier.fit(X_train, y_train)

# Predict on test dataset
y_predictions = rf_classifier.predict(X_test)

print(classification_report(y_test, y_predictions,
      target_names=['ground', 'vegetation', 'buildings']))

# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# axs[0].scatter(X_test['X'], X_test['Y'], c=y_test, s=0.05)
# axs[0].set_title('3D Point Cloud Ground Truth')
# axs[1].scatter(X_test['X'], X_test['Y'], c=y_predictions, s=0.05)
# axs[1].set_title('3D Point Cloud Predictions')
# axs[2].scatter(X_test['X'], X_test['Y'], c=y_test-y_predictions,
#                cmap=plt.cm.rainbow, s=0.5*(y_test-y_predictions))
# axs[2].set_title('Differences')

# Predict on validation dataset
val_features_test_scaled = MinMaxScaler().fit_transform(val_features_test)
val_predictions = rf_classifier.predict(val_features_test_scaled)

print(classification_report(val_labels_test, val_predictions,
      target_names=['ground', 'vegetation', 'buildings']))

# Save the trained model for later usage
pickle.dump(rf_classifier, open(data_folder+"urban_classifier.gox", 'wb'))
