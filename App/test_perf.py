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
# from skimage.filters import sobel


def cases_zernike():
    for size in [64, 128, 256, 512]:
        tags = {'size': (size, size)}
        img = imread('lena128x128.jpg')
        img_resized = resize(img, (size, size))
        yield neurtu.delayed(zernike_moment, tags=tags)(img_resized, 15)


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
    coef = fourier1_color(img)

    return coef


def read_compute_hu(f):
    img = imread(f)
    coef = hu_moment_color(img)

    return coef


def read_compute_zernike(f):
    img = imread(f)
    coef = zernike_moment_color(img, 15)

    return coef


def extract_class_from_path(s):
    try:
        found = re.search('obj(.+?)/', s).group(1)
    except AttributeError:
        found = ''

    return found


def test_perf_fourier_svm():
    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-100', 'obj*', '*.png')
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

    return scores


def test_perf_hu_svm():
    Y = []

# read the data
    # path_data = os.path.join('..', 'Dataset', 'coil-20-proc', 'obj*', '*.png')
    path_data = os.path.join('..', 'Dataset', 'coil-100', 'obj*', '*.png')
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

    return scores


def test_perf_zernike_svm():
    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-100', 'obj*', '*.png')
    filenames = sorted(glob.glob(path_data))
    data = Parallel(n_jobs=-1)(delayed(
        read_compute_zernike)(f) for f in filenames)

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

    return scores


def timing_zernike():
    df = neurtu.timeit(cases_zernike())
    df['wall_time'] *= 1000
    print(df)


def timing_hu():
    df = neurtu.timeit(cases_hu())
    df['wall_time'] *= 1000
    print(df)


def timing_fourrier1():
    df = neurtu.timeit(cases_fourier1())
    df['wall_time'] *= 1000
    print(df)


def debug_zernike(verbose=False):
    img = imread('lena128x128.jpg')
    img_resized = resize(img, (128, 128))
    coef_zer = zernike_moment(img_resized, 15)

    if verbose is True:
        plt.figure()
        plt.plot(coef_zer)
        plt.show()


def main():
    # manip timing
    # timing Fourier1
    # timing_fourrier1()

    # timing Hu
    # timing_hu()

    # timing Zernike
    # timing_zernike()

    # Test performances Fourier
    s_f = test_perf_fourier_svm()

    # Test performances Fourier
    s_h = test_perf_hu_svm()

    # Test performances Zernike
    s_z = test_perf_zernike_svm()

    # debug Zernike
    # debug_zernike(verbose=True)

    v_svc_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ind = np.arange(len(s_f))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/3, s_f['test_score'], width,
                    color='SkyBlue', label='Fourier1')
    rects2 = ax.bar(ind + width/3, s_h['test_score'], width,
                    color='IndianRed', label='Hu')
    rects3 = ax.bar(ind + width/3, s_z['test_score'], width,
                    label='Zernike')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores with 3 methods of characterization')
    ax.set_xticks(ind)
    ax.set_xticklabels(('0.001, 0.01, 0.1, 1, 10, 100, 1000'))
    ax.legend()
    plt.savefig('comparatif.pdf')
    plt.show()


if __name__ == '__main__':
    main()
