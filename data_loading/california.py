import sklearn.datasets as skl_ds
from sklearn import preprocessing as skpp
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
import torch
import numpy as np

import utils

MASK_LIMITS = torch.tensor([
    [[-0.6138,-0.0247], [-0.5402,0.1491]],
    [[-0.4063, -0.141], [-0.0995, 0.154]],
    [[0.2202, -0.8749], [0.3291, -0.7641]],
    ])

OUTLIER_FRACTION = 0.005 # Assumed fraction of outliers in the data

def load_cal(args):
    raw_X, y = skl_ds.fetch_california_housing(return_X_y=True)
    y = np.expand_dims(y, axis=1)
    N_ds = raw_X.shape[0]

    # Split features and coordinates
    long_lat = np.stack([raw_X[:,7], raw_X[:,6]], axis=1) # Shape (N, 2)
    X_features_raw = raw_X[:,:6]

    # Transform features
    # Same feature transformation as in original paper by Pace and Barry
    y = np.log(y)
    X_features = np.stack((
        X_features_raw[:,0],
        np.power(X_features_raw[:,0], 2),
        np.power(X_features_raw[:,0], 3),
        np.log(X_features_raw[:,1]),
        np.log(X_features_raw[:,2]),
        np.log(X_features_raw[:,3]),
        np.log(X_features_raw[:,5]),
        np.log(X_features_raw[:,4]) - np.log(X_features_raw[:,5]), # ln(households)
        ), axis = 1
    )

    # Remove outliers
    n_features = X_features_raw.shape[1]
    ellipse = EllipticEnvelope(contamination=OUTLIER_FRACTION, random_state=42)
    # Perform outlier detection on original features (more accurate in practice)
    outlier_labels = ellipse.fit_predict(X_features_raw)
    inlier_filter = (outlier_labels == 1)
    n_outliers = np.sum((outlier_labels == -1).astype(float))

    if args.plot:
        fig, axes = plt.subplots(nrows=np.ceil(n_features/3).astype(int),
                ncols=3, figsize=(20,20))
        axes = axes.flatten()
        for i in range(n_features):
            x = X_features[:,i]
            ax = axes[i]
            ax.scatter(x, y[:,0], s=1, c=outlier_labels)
            ax.set(title="Feature {}".format(i), xlabel="x_i", ylabel="y")

        fig.suptitle("Outliers")
        plt.show()

    print("Dataset size: {}".format(N_ds))
    print("Removing {} outliers".format(n_outliers))

    X_features = X_features[inlier_filter]
    y = y[inlier_filter]
    long_lat = long_lat[inlier_filter]

    # Standardize features and y
    scaler = skpp.MinMaxScaler()
    X_features = scaler.fit_transform(X_features)
    y = scaler.fit_transform(y)

    # Project longitude and latitude
    pos = utils.project_eqrect(long_lat)

    return X_features, y, pos, MASK_LIMITS

