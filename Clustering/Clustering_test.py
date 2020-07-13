
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_samples    = 1500
random_state = 170
X_varied, y_varied = make_blobs(
    n_samples     = n_samples,
    cluster_std   = [1.0, 2.5, 0.5],
    random_state  = random_state
)

transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_varied = np.dot(X_varied, transformation)


from Clustering import FuzzyCMeanCluster
model = FuzzyCMeanCluster(
    n_cluster = 3, 
    n_iter    = 50, 
    m         = 2
)

model.Fit(X_varied)
y_pred = model.Predict(X_varied)


plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.scatter(
    X_varied[:, 0], X_varied[:, 1], 
    c = y_pred, 
    s = 2
)
plt.title("Fuzzy C Mean (m=2)")

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
plt.subplot(122)
plt.scatter(
    X_varied[:, 0], X_varied[:, 1], 
    c = y_pred, s = 2
)
plt.title("K Mean")