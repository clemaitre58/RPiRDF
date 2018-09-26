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


def cases():
    for size in [65, 129, 257, 513]:
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

    ###########################################################################
    # IO section:
    # 1. Read all available data
    # 2. Remove the files which do not contain all required information

    # cache the reading if we need to execute the script again
    # memory = Memory(cachedir=os.path.join('cache', 'bikereadcache'))
    # bikeread_cached = memory.cache(bikeread, verbose=1)

    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-20-proc', 'obj*', '*.png')
    filenames = sorted(glob.glob(path_data))
    data = Parallel(n_jobs=-1)(delayed(
        read_compute_fourrier1)(f) for f in filenames)

    for f in filenames:
        Y.append(int(extract_class_from_path(f)))
    Y = np.array(Y)

# # filter the activity which do not contain the required information
# fields = ['elevation', 'cadence', 'distance', 'heart-rate', 'power', 'speed']
# valid_data = []
# for activity in data:
#     if set(fields).issubset(activity.columns):
#         if not pd.isnull(activity).any().any():
#             valid_data.append(activity)
# data = valid_data

###############################################################################
# Data processing
# 1. Compute extra information: acceleration, gradient for elevation and
# heart-rate. and compute gradient over 5 seconds.
# 2. Use a gradient boosting as estimator.
# 3. Make a cross-validation to obtain true estimate of the score.
# 4. Repeat the experiment to get the prediction for visualization purpose.

# for activity_idx in range(len(data)):
#     # compute acceleration
#     data[activity_idx] = acceleration(data[activity_idx])
#     # compute gradient elevation
#     data[activity_idx] = gradient_elevation(data[activity_idx])
#     # compute gradient heart-rate
#     data[activity_idx] = gradient_heart_rate(data[activity_idx])
#     # compute the gradient information over 10 sec for the some fields
#     fields = ['elevation', 'cadence', 'heart-rate', 'speed',
#               'gradient-elevation', 'gradient-heart-rate', 'acceleration']
#     data[activity_idx] = gradient_activity(data[activity_idx],
#                                            periods=range(1, 6),
#                                            columns=fields)
#
# for#  activity in data:
#     activity.replace([np.inf, -np.inf], np.nan, inplace=True)
#
# data_concat = pd.concat(data)
# y = data_concat['original']['power']
# X = data_concat.drop('power', axis=1, level=1)
# X.fillna(X.mean(), inplace=True)
# groups = []
# for group_idx, activity in enumerate(data):
#     groups += [group_idx] * activity.shape[0]
# groups = np.array(groups)
    X = np.array(data)
    print(len(data))
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

    # Store the prediction for visualization
    # y_pred = cross_val_predict(
    #     GradientBoostingRegressor(random_state=42, n_jobs=-1),
    #     X, Y, groups=None,
    #     cv=GroupKFold(n_splits=3), n_jobs=1)
    # path_results = os.path.join('results', 'machine_learning_model')
    # if not os.path.exists(path_results):
    #     os.makedirs(path_results)
    # f = os.path.join(path_results, 'y_pred.csv')
    # pd.Series(y_pred, index=y.index).to_csv(f)
    # f = os.path.join(path_results, 'y_true.csv')
    # y.to_csv(f)
    # np.save(os.path.join(path_results, 'groups.npy'), groups)


def test_perf_hu_svm():

    ###########################################################################
    # IO section:
    # 1. Read all available data
    # 2. Remove the files which do not contain all required information

    # cache the reading if we need to execute the script again
    # memory = Memory(cachedir=os.path.join('cache', 'bikereadcache'))
    # bikeread_cached = memory.cache(bikeread, verbose=1)

    Y = []

# read the data
    path_data = os.path.join('..', 'Dataset', 'coil-20-proc', 'obj*', '*.png')
    filenames = sorted(glob.glob(path_data))
    data = Parallel(n_jobs=-1)(delayed(
        read_compute_hu)(f) for f in filenames)

    for f in filenames:
        Y.append(int(extract_class_from_path(f)))
    Y = np.array(Y)

# # filter the activity which do not contain the required information
# fields = ['elevation', 'cadence', 'distance', 'heart-rate', 'power', 'speed']
# valid_data = []
# for activity in data:
#     if set(fields).issubset(activity.columns):
#         if not pd.isnull(activity).any().any():
#             valid_data.append(activity)
# data = valid_data

###############################################################################
# Data processing
# 1. Compute extra information: acceleration, gradient for elevation and
# heart-rate. and compute gradient over 5 seconds.
# 2. Use a gradient boosting as estimator.
# 3. Make a cross-validation to obtain true estimate of the score.
# 4. Repeat the experiment to get the prediction for visualization purpose.

# for activity_idx in range(len(data)):
#     # compute acceleration
#     data[activity_idx] = acceleration(data[activity_idx])
#     # compute gradient elevation
#     data[activity_idx] = gradient_elevation(data[activity_idx])
#     # compute gradient heart-rate
#     data[activity_idx] = gradient_heart_rate(data[activity_idx])
#     # compute the gradient information over 10 sec for the some fields
#     fields = ['elevation', 'cadence', 'heart-rate', 'speed',
#               'gradient-elevation', 'gradient-heart-rate', 'acceleration']
#     data[activity_idx] = gradient_activity(data[activity_idx],
#                                            periods=range(1, 6),
#                                            columns=fields)
#
# for activity in data:
#     activity.replace([np.inf, -np.inf], np.nan, inplace=True)
#
# data_concat = pd.concat(data)
# y = data_concat['original']['power']
# X = data_concat.drop('power', axis=1, level=1)
# X.fillna(X.mean(), inplace=True)
# groups = []
# for group_idx, activity in enumerate(data):
#     groups += [group_idx] * activity.shape[0]
# groups = np.array(groups)
    X = np.array(data)
    print(len(data))
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

    # Store the prediction for visualization
    # y_pred = cross_val_predict(
    #     GradientBoostingRegressor(random_state=42, n_jobs=-1),
    #     X, Y, groups=None,
    #     cv=GroupKFold(n_splits=3), n_jobs=1)
    # path_results = os.path.join('results', 'machine_learning_model')
    # if not os.path.exists(path_results):
    #     os.makedirs(path_results)
    # f = os.path.join(path_results, 'y_pred.csv')
    # pd.Series(y_pred, index=y.index).to_csv(f)
    # f = os.path.join(path_results, 'y_true.csv')
    # y.to_csv(f)
    # np.save(os.path.join(path_results, 'groups.npy'), groups)


def main():
    # manip timing

    # df = neurtu.timeit(cases())
    # df['wall_time'] *= 1000
    # print(df)
    # img = imread('lena128x128.jpg')
    # img_resized = resize(img, (152, 324))
    # coef_desc = fourier1(img_resized)
    # print(coef_desc)

    # Test performances Fourier
    # test_perf_fourier_svm()

    # Test performances Fourier
    test_perf_hu_svm()


if __name__ == '__main__':
    main()
