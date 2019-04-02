# importing necessary libraries
import warnings
warnings.filterwarnings('ignore')

import random
random.seed(42)

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import matplotlib.pyplot as plt
# %matplotlib inline


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def readDataframe():
    df = pd.read_csv('../input/ads_hour.csv',index_col=['Date'], parse_dates=['Date'])
    # with plt.style.context('bmh'):
    #     plt.figure(figsize=(15, 8))
    #     plt.title('Ads watched (hour ticks)')
    #     plt.plot(df.ads)
    #     plt.show()
    return  df


def prepareData(data, lag_start=5, lag_end=14, test_size=0.15):
    """
    series: pd.DataFrame
        dataframe with timeseries

    lag_start: int
        initial step back in time to slice target variable
        example - lag_start = 1 means that the model
                  will see yesterday's values to predict today

    lag_end: int
        final step back in time to slice target variable
        example - lag_end = 4 means that the model
                  will see up to 4 days back in time to predict today

    test_size: float
        size of the test dataset after train/test split as percentage of dataset

    """
    data = pd.DataFrame(data.copy())
    data.columns = ["y"]

    # calculate test index start position to split data on train test
    test_index = int(len(data) * (1 - test_size))

    # adding lags of original time series data as features
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    # transforming df index to datetime and creating new variables
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday

    # since we will be using only linear models we need to get dummies from weekdays
    # to avoid imposing weird algebraic rules on day numbers
    data = pd.concat([
        data.drop("weekday", axis=1),
        pd.get_dummies(data['weekday'], prefix='weekday')
    ], axis=1)

    # encode hour with sin/cos transformation
    # credits - https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data.drop(["hour"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    # splitting whole dataset on train and test
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test


def plotModelResults(model, df_train, df_test, y_train, y_test, cv, plot_intervals=False, scale=1.96):
    """
    Plots modelled vs fact values

    model: fitted model

    df_train, df_test: splitted featuresets

    y_train, y_test: targets

    plot_intervals: bool, if True, plot prediction intervals

    scale: float, sets the width of the intervals

    cv: cross validation method, needed for intervals

    """
    # making predictions for test
    prediction = model.predict(df_test)

    plt.figure(figsize=(20, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        # calculate cv scores
        cv = cross_val_score(
            model,
            df_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error"
        )

        # calculate cv error deviation
        deviation = np.sqrt(cv.std())

        # calculate lower and upper intervals
        lower = prediction - (scale * deviation)
        upper = prediction + (scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

    # calculate overall quality on test set
    mae = mean_absolute_error(prediction, y_test)
    mape = mean_absolute_percentage_error(prediction, y_test)
    plt.title("MAE {}, MAPE {}%".format(round(mae), round(mape, 2)))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def getCoefficients(model, X_train):
    """Returns sorted coefficient values of the model"""
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    return coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

def plotCoefficients(model, X_train):
    """Plots sorted coefficient values of the model"""
    coefs = getCoefficients(model, X_train)

    plt.figure(figsize=(20, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    plt.show()

    ax = sns.heatmap(X_train)
    plt.show()

def cross_validate (model, X_train_scaled, X_test_scaled, y_train, y_test):
    tscv = TimeSeriesSplit(n_splits = 5)
    scores = cross_val_score(model, X_train_scaled, y_train, scoring='neg_mean_absolute_error', cv=tscv)
    mean_score = -np.mean(scores)
    # plotModelResults(model, X_train_scaled, X_test_scaled, y_train, y_test, cv=tscv)
    return mean_score

def main ():
    df = readDataframe();
    X_train, X_test, y_train, y_test = prepareData(df, 12, 48, 0.3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Q1
#     lin_reg = LinearRegression()
#     lin_reg.fit(X_train_scaled, y_train)
#
#     score = cross_validate(lin_reg, X_train_scaled, X_test_scaled, y_train, y_test)
#     print('Question 1.: {}'.format(score)) # 4490.0642733984405
#     plotCoefficients(lin_reg, X_train)

# Q2
    lassocv = LassoCV()
    lassocv.fit(X_train_scaled, y_train)
    score = cross_validate(lassocv, X_train_scaled, X_test_scaled, y_train, y_test)
    print('LassoCV scores: {}'.format(score))
    # plotCoefficients(lassocv, X_train)

    coefs = getCoefficients(lassocv, X_train)
    coefs['zeros'] = (np.abs(coefs['coef']) < 0.0001).astype('int64')
    zerosCount = np.sum(coefs['zeros'])
    print('Question 2. Zero params = {}'.format(zerosCount)) # 17




if __name__ == '__main__':
    main()
