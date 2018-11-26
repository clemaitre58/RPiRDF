import neurtu
import os
import glob
import re

# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from joblib import Parallel
from joblib import delayed
# from joblib import Memory

# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_predict

# from xgboost import XGBRegressor as GradientBoostingRegressor
from sklearn.svm import SVC

from skimage.io import imread
from skimage.transform import resize

from DescGlob import fourier1
from DescGlob import fourier1_color
from DescGlob import hu_moment
from DescGlob import hu_moment_color
from DescGlob import zernike_moment
from DescGlob import zernike_moment_color


img = imread('../astronaut.png')
img_resized = resize(img, (128, 128))
coef = zernike_moment_color(img_resized, 15)
print(len(coef))
print(type(coef))
print(coef)
