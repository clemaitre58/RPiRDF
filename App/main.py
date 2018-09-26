import neurtu
import os
import glob
import re

# import pandas as pd
import numpy as np

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
from DescGlob import hu_moment
# from skimage.filters import sobel


def cases_hu():
    for size in [64, 128, 256, 512]:
        tags = {'size': (size, size)}
        img = imread('lena128x128.jpg')
        img_resized = resize(img, (size, size))
        yield neurtu.delayed(hu_moment, tags=tags)(img_resized)


def cases_fourier1():
    for size in [64, 128, 256, 512]:
        tags = {'size': (size, size)}
        img = imread('lena128x128.jpg')
        img_resized = resize(img, (size, size))
        yield neurtu.delayed(fourier1, tags=tags)(img_resized)


def read_compute_fourrier1(f):
    img = imread(f)
    coef = fourier1(img)

    return coef


def read_compute_hu(f):
    img = imread(f)
    coef = hu_moment(img)

    return coef


def read_compute_fourrier1_col(f):
    img_rgb = imread(f)
    coef_r = fourier1(img_rgb[:][:][0])
    coef_g = fourier1(img_rgb[:][:][1])
    coef_b = fourier1(img_rgb[:][:][2])

    return coef_r + coef_g + coef_b


def extract_class_from_path(s):
    try:
        found = re.search('obj(.+?)/', s).group(1)
    except AttributeError:
        found = ''

    return found


def test_perf_fourier_svm():
    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-20-proc', 'obj*', '*.png')
    filenames = sorted(glob.glob(path_data))
    data = Parallel(n_jobs=-1)(delayed(
        read_compute_fourrier1)(f) for f in filenames)

    for f in filenames:
        Y.append(int(extract_class_from_path(f)))
    Y = np.array(Y)

    X = np.array(data)
    clf = make_pipeline(StandardScaler(), SVC(random_state=42, gamma='auto'))
    grid = GridSearchCV(clf,
                       param_grid={'svc__C': [0.001, 0.01, 0.1, 1, 10,
                                              100, 1000]},
                       cv=5, iid=False)
    scores = cross_validate(grid,
                            X, Y,
                            cv=5, n_jobs=-1,
                            return_train_score=True,
                            verbose=0)

    print('The obtained scores on training and testing in terms of '
          'accuracy: \n')
    print(scores)


def test_perf_hu_svm():
    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-20-proc', 'obj*', '*.png')
    filenames = sorted(glob.glob(path_data))
    data = Parallel(n_jobs=-1)(delayed(
        read_compute_hu)(f) for f in filenames)

    for f in filenames:
        Y.append(int(extract_class_from_path(f)))
    Y = np.array(Y)
    X = np.array(data)
    clf = make_pipeline(StandardScaler(), SVC(random_state=42, gamma='auto'))
    grid = GridSearchCV(clf,
                       param_grid={'svc__C': [0.001, 0.01, 0.1, 1, 10,
                                              100, 1000]},
                       cv=5, iid=False)
    scores = cross_validate(grid,
                            X, Y,
                            cv=5, n_jobs=-1,
                            return_train_score=True,
                            verbose=0)

    print('The obtained scores on training and testing in terms of '
          'accuracy: \n')
    print(scores)


def timing_hu():
    df = neurtu.timeit(cases_hu())
    df['wall_time'] *= 1000
    print(df)


def timing_fourrier1():
    df = neurtu.timeit(cases_fourier1())
    df['wall_time'] *= 1000
    print(df)


def main():
    # manip timing
    # timing Fourier1
    timing_fourrier1()

    # timing hu
    timing_hu()

    # Test performances Fourier
    # test_perf_fourier_svm()

    # Test performances Fourier
    # test_perf_hu_svm()


if __name__ == '__main__':
    main()
