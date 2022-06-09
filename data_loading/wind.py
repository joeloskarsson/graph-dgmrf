import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import PolynomialFeatures

import utils
import constants

MASK_LIMITS = torch.tensor([
    [[-0.818,0.203], [-0.398,0.378]],
    [[0.364,-0.161], [0.638,-0.045]],
    [[-0.561,-0.329], [-0.064,-0.149]],
    ])

DATA_FILE = "wtk_site_metadata.csv"
LAT_LIMIT = 24.4 # Limit of continental US

def load_wind_speed(args):
    return load_wind(True)

def load_wind_cap(args):
    return load_wind(False)

def load_wind(return_speed):
    data_path = os.path.join(constants.RAW_DATA_DIR, DATA_FILE)
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    long_lat = data[:,1:3]
    wind_speed = data[:, 8]
    cap_factor = data[:, 9]

    # Mask away points south of US
    lat_mask = (long_lat[:,1] > LAT_LIMIT)
    long_lat = long_lat[lat_mask]
    wind_speed = wind_speed[lat_mask]
    cap_factor = cap_factor[lat_mask]

    pos = utils.project_eqrect(long_lat)

    if return_speed:
        y = wind_speed
    else:
        y = cap_factor

    # Polynomial features based on position
    poly_feat = PolynomialFeatures(degree=2, include_bias=False)
    X_features = poly_feat.fit_transform(pos)

    return X_features, y, pos, MASK_LIMITS

