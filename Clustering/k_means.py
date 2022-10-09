from curses import KEY_LEFT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data_folder = "/home/goksan/Downloads/drive-download-20220910T084041Z-001/"
dataset = "KME_planes.xyz"

### Extract the data ###
x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(
    data_folder+dataset, skiprows=1, delimiter=";", unpack=True)

# plt.subplot(1, 2, 1)
# plt.scatter(x, z, c=intensity, s=0.05)
# plt.axhline(y=np.mean(z), color='r', linestyle='-')
# plt.title('First view')
# plt.xlabel('X-axis')
# plt.ylabel('Z-axis')

# plt.subplot(1, 2, 2)  # index 2
# plt.scatter(y, z, c=intensity, s=0.05)
# plt.axhline(y=np.mean(z), color='r', linestyle='-')
# plt.title('Second view')
# plt.xlabel('Y-axi')
# plt.ylabel('Z-axis')

# plt.show()

### Filter out the ground ###
pcd = np.column_stack((x, y, z))
mask = z > np.mean(z)
spatial_query = pcd[mask]
# ax = plt.axes(projection='3d')
# ax.scatter(x[mask], y[mask], z[mask], c=intensity[mask], s=0.1)
# ax.scatter(x[mask], y[mask], c=intensity[mask], s=0.1)

### Clustering with "n" determined by Elbow's method ###
X = np.column_stack((x[mask], y[mask]))
# Choosing # clusters with Elbow method
# wcss = []  # Within cluster sum of square
# for i in range(1, 20):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1, 20), wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

### Clustering more than geometric features ###
X = np.column_stack((x[mask], y[mask], z[mask],
                    illuminance[mask], nb_of_returns[mask], intensity[mask]))
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], c='r', s=15.0, marker='o')
# Note that the cluster centers does not geometrically make sense anymore
plt.show()
