import math
from sklearn import preprocessing
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense
from keras.layers import LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import io
from datetime import datetime, timedelta
import re
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score

### GET DATA ##################################################################
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}                                                   # header illusion...

def current_GLD():
    """Returns float USD amount of SPDR GLD ETF from business insider.com"""
    url = 'http://markets.businessinsider.com/etfs/spdr-gold-shares'
    my_req = requests.get(url, headers=header)
    my_soup = bs(my_req.content, "html.parser")
    t = my_soup.find_all('span', {"class": "aktien-big-font text-nowrap"})
    val_data = re.findall("\d+\.\d+", str(t))       # extract digits
    return float(val_data[0])                       # convert to float

def get_SPDRGLD():
    """Returns pandas df with datetime index, GLD price, and LMBA prices"""
    url = 'http://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv'
    my_req = requests.get(url, headers=header)          # pull down csv from sp
    gld_hist = pd.read_csv(io.StringIO(my_req.content.decode('utf-8')),
                           skiprows=6,                  # clean up columns
                           parse_dates=True,
                           index_col="Date")
    gld_hist.index.rename('date', inplace=True)         # match index to cccagg

    gld_hist = gld_hist[[' GLD Close', ' LBMA Gold Price',
                         ' Total Net Asset Value in the Trust']]
                                                        # strip out $ sign
    gld_hist.loc[:, gld_hist.columns.values[1]] = \
        gld_hist[gld_hist.columns[1]].str.replace('$', '')

                                                        # NA replace + ffill
    gld_hist.replace(' HOLIDAY', np.NaN, inplace=True)
    gld_hist.replace(' NYSE Closed', np.NaN, inplace=True)
    gld_hist.replace(' AWAITED', np.NaN, inplace=True)
    gld_hist = gld_hist.fillna(method='ffill')

    gld_hist = gld_hist.rename(columns={            # clean column names
        gld_hist.columns.values[0]: 'GLD_close',
        gld_hist.columns.values[1]: 'LMBA_price',
        gld_hist.columns.values[2]: 'GLD_market_cap'})
                                                    # convert to float values
    gld_hist["GLD_close"] = gld_hist.GLD_close.astype(float)
    gld_hist["LMBA_price"] = gld_hist.LMBA_price.astype(float)
    gld_hist["GLD_market_cap"] = gld_hist.GLD_market_cap.astype(float)

                                                    # get current GLD prices
    new_entry = pd.DataFrame.from_dict({pd.to_datetime(datetime.now()): {
        "GLD_close": current_GLD(),                 # reuse yestdy's LMBA price
        "LMBA_price": gld_hist.iloc[-1][1],
        "GLD_market_cap": gld_hist.iloc[-1][2]
    }}, orient='index')
    new_entry.index.rename('date', inplace=True)    # name index like gld_hist
    return gld_hist.append(new_entry)               # add new_entry, return

def get_sp500():
    """ Pull Daily Historical S&P500 from FRED https://fred.stlouisfed.org"""
    link = "https://fred.stlouisfed.org/data/SP500.txt"
    my_req = requests.get(link, headers=header)
    sp500 = pd.read_table(io.StringIO(my_req.content.decode('utf-8')),
                          parse_dates=True,
                          index_col='DATE',
                          dtype={'VALUE': np.float64},
                          delim_whitespace=True,
                          na_values='.',
                          skiprows=35)
    sp500.rename(columns = {'VALUE':'SP500'}, inplace=True)
    sp500 = sp500.reindex(sp500.index.rename('date'))
    return(sp500)

def get_crypto(symb, yrs):
    """ for pulling 'yrs' of data from Cryptocompare API"""
    c_daily = "https://min-api.cryptocompare.com/data/histoday"
    param = {'fsym': symb, 'tsym': 'USD',
             'e': 'CCCAGG', 'limit': round(yrs*365)}    # x years
    t = requests.get(c_daily, param).json()['Data']
    ck = pd.DataFrame.from_dict(t)
    date_in = pd.to_datetime(ck.time, unit='s',
                             origin='unix')
    ck['date'] = date_in
    ck.index = date_in
    ck.index.rename('date', inplace=True)
    ck = ck.query('volumefrom != 0')
    ck = ck[['close', 'volumeto']]
    ck = ck.assign(btc_close = ck.close)
    ck = ck.assign(btc_volume =ck.volumeto)
    return ck[['btc_close', 'btc_volume']]

def get_volume():
    """For obtaining crypto volume"""
    cols = ['date', 'timestamp', 'volume']
    limit = 1800
    url = "https://min-api.cryptocompare.com/data/exchange/histoday?tsym=USD&limit=" + \
          str(limit)
    t = requests.get(url).json()['Data']
    data = pd.DataFrame.from_dict(t)
    date_in = pd.to_datetime(data.time, unit='s',
                             origin='unix')
    data['date'] = date_in
    data.index = date_in
    data.index.rename('date', inplace=True)
    data = data.assign(crypto_volume = data.volume)
    return data[['crypto_volume']]

def combined_data(coin, yrs):
    """Combines features with crypto data"""
    cry = get_crypto(coin, yrs)
    gld = get_SPDRGLD()
    sp5 = get_sp500()
    vol = get_volume()
    df = cry.join(gld,how='outer').fillna(method='ffill')
    df = df.join(sp5, how='outer').fillna(method='ffill')
    df = df.join(vol, how ='outer').fillna(method='ffill')
    df = df.dropna()
    return df

### FEATURE ENGINEERING  ######################################################
def rate_of_change(data):
    # rate of change
    data["btc_close_roc"] = np.gradient(data['btc_close'])
    data["btc_volume_roc"] = np.gradient(data['btc_volume'])
    data["crypto_volume_roc"] = np.gradient(data['crypto_volume'])
    data["gold_close_roc"] = np.gradient(data['GLD_close'])
    data["lmba_gold_roc"] = np.gradient(data['LMBA_price'])
    data["gold_volume_roc"] = np.gradient(data['GLD_market_cap'])
    data["SP500_roc"] = np.gradient(data['SP500'])
    return data

def parse_date(data):
    # parse date into year, month, week, day, dayofweek
    data["year"] = data.index.year
    data['month'] = data.index.month
    data['week'] = data.index.week
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    return data

def get_dummies(data):
    # day of week
    dayofweek_dummies = pd.get_dummies(data['dayofweek'], prefix='dayofweek')
    data = pd.concat([data, dayofweek_dummies], axis=1)

    # year
    year_dummies = pd.get_dummies(data['year'], prefix='year')
    data = pd.concat([data, year_dummies], axis=1)
    # month
    month_dummies = pd.get_dummies(data['month'], prefix='month')
    data = pd.concat([data, month_dummies], axis=1)
    data.drop('year', axis=1, inplace=True)
    data.drop('month', axis=1, inplace=True)
    data.drop('dayofweek', axis=1, inplace=True)
    return (data)

def add_feats(x):
    """Function to add all kinds of calculated columns"""
    # 1-day Log Returns
    x = x.assign(gld_lr=np.log(x.GLD_close / x.GLD_close.shift(1)))
    x = x.assign(gmc_lr=np.log(x.GLD_market_cap / x.GLD_market_cap.shift(1)))
    x = x.assign(sp5_lr=np.log(x.SP500 / x.SP500.shift(1)))
    x = x.assign(crv_lr=np.log(x.crypto_volume / x.crypto_volume.shift(1)))

    # BTC Stats...
    x = x.assign(btcMA3=x.rolling(3).mean().btc_close)  # rolling mean, std
    x = x.assign(btcMA5=x.rolling(5).mean().btc_close)
    x = x.assign(btcMA10=x.rolling(10).mean().btc_close)
    x = x.assign(btcMA20=x.rolling(20).mean().btc_close)
    # x = x.assign(btcSD5 = x.rolling(5).std().btc_close)

    # Gold Stats...
    x = x.assign(gldMA2=x.rolling(2).mean().GLD_close)
    x = x.assign(gldMA5=x.rolling(5).mean().GLD_close)
    x = x.assign(gldMA10=x.rolling(10).mean().GLD_close)
    x = x.assign(gldMA20=x.rolling(20).mean().GLD_close)
    # x = x.assign(gldSD5 = x.rolling(5).std().GLD_close)

    # response vars...
    x = x.assign(btc_lr=np.log(x.btc_close / x.btc_close.shift(1)))
    x = x.assign(btv_lr=np.log(x.btc_volume / x.btc_volume.shift(1)))
    x = x.assign(btc_lg1p=np.log1p(x.btc_close)).dropna()  # log of price
    return(x)

def get_combined_data():
    """Function that adds all columns for model-building"""
    data = combined_data('BTC', 5)
    # static = data.copy()
    # data = static.copy()
    data = parse_date(data)
    data = get_dummies(data)
    data = rate_of_change(data)
    data = add_feats(data)
    return data

### DIAGNOSTICS ###############################################################
def rmse(actual, predict):
    """Root Mean Square Error"""
    predict = np.array(predict)
    actual = np.array(actual)
    mse = mean_squared_error(actual, predict)
    rmse = sqrt(mse)
    return rmse

def rmsle(actual, predict):
    """Root Mean Square Log Error"""
    predict = np.array(predict)
    actual = np.array(actual)

    log_predict = np.log(predict + 1)
    log_actual = np.log(actual + 1)

    diff = log_predict - log_actual
    square_diff = np.square(diff)

    msd = square_diff.mean()
    score = np.sqrt(msd)
    return score

def my_crossval(rfm, X_train, y_train, m_name=""):
    """Score the models"""
    # feature importance
    # importance = sorted(zip(map(lambda x: round(x, 4), rfm.feature_importances_),
    #                  feature_names), reverse=True)
    rf_cv_rmsle_score = cross_val_score(rfm, X_train, y_train ,
                                        cv = 5, scoring = rmsle_score).mean()
    # rf_cv10_rmsle_score = cross_val_score(rfm, X_train, y_train,
    #                                       cv=10,scoring=rmse_score).mean()
    pred = rfm.predict(X=X_train)
    rf_rmse = rmse(y_train, pred)

    print("RMSLE CV Score: {0:.5f}".format(rf_cv_rmsle_score))
    # print("RMSLE CV20 Score: {0:.5f}".format(rf_cv10_rmsle_score))
    print("RMSE Value: {0:.5f}".format(rf_rmse))
    scores = {m_name: {"RMSLE": rf_cv_rmsle_score,
                       # "RMSLE CV10": rf_cv10_rmsle_score,
                       "RMSE": rf_rmse}}
    score_df = pd.DataFrame.from_dict(scores, orient='index')
    return score_df

### PLOTS #####################################################################

def tsplot(y, lags=None, figsize=(8, 6), n = ""):
    """ Plot for reviewing stationarity (or lack thereof) in a series
    Mix of plots from DataCamp and http://www.blackarbs.com/"""
    fig = plt.figure(figsize=figsize)
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Plots %s' % n)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()
    plt.close()
    return

def act_v_predict_plot(rfm_preds, y_test, days):
    """Plot time-series, in order of prediction vs actual"""
    plt.clf()
    rvw_results = pd.DataFrame(
        {"actual": y_test.values, "predicted": rfm_preds}, index=y_test.index)

    rvw_results.sort_index(inplace=True)
    n_days = days
    smaller = rvw_results[
              (datetime.utcnow().date() - timedelta(days=n_days)).isoformat():]
    plt.plot(smaller.index, smaller.actual, '-', color='blue', label="Actual")
    plt.plot(smaller.index, smaller.predicted, '--', color='green',
             label=('Predicted'))
    plt.legend(['Actual', 'Predicted'], loc='best')
    plt.grid('on', which='major', linestyle='--', alpha=.5, color='gray')
    plt.title("Actual vs. Prediction: {0} days".format(n_days))
    plt.xticks(rotation=30)
    plt.show()

def pred_vs_act_sca(preds, y_test):
    """A plot to review how prediction compares w/ actual via scatter"""
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y_test, preds, c='b', marker='x', label='Actual', alpha =.5)
    ax.scatter(preds, y_test, c='g', marker='s', label='Predicted', alpha =.5)
    plt.legend(loc='best')
    plt.title('Actual vs. Predicted')
    plt.show()

### MODEL PREPERATION #########################################################
def label_and_feat(data, tag):
    """for assigning response + features for models"""
    if tag == 'r':
        label_name = 'btc_close'
        feature_names = list(data.columns[1:len(data.columns)])
    else:
        label_name = 'btc_close'
        feature_names = list(data.columns[2:len(data.columns)-3])
    print("*" * 75)
    print("Modeling: {0}".format(label_name))
    print("Features Include:")
    for i in range(round(len(feature_names) / 10) + 1):
        print(", ".join(feature_names[i * 10:(i + 1) * 10]))
    print("*" * 75)
    return label_name, feature_names

def prep_data(data, tag):
    """Select columns of interest for 3 models"""
    # Select features, response
    if tag == 'j':
        col_nam = ['btc_close', 'btc_volume', 'GLD_close', 'GLD_market_cap',
                   'SP500','crypto_volume', 'week', 'day',
                   'dayofweek_0', 'dayofweek_1', 'dayofweek_2', 'dayofweek_3',
                   'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'year_2013',
                   'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
                   'month_1', 'month_2', 'month_3', 'month_4',
                   'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
                   'month_10', 'month_11', 'month_12',
                   'gld_lr', 'gmc_lr', 'sp5_lr', 'crv_lr', 'gldMA2', 'gldMA5',
                   'gldMA10','gldMA20','btcMA3', 'btcMA5','btcMA10','btcMA20',
                   'btc_lr','btv_lr', 'btc_lg1p']
    elif tag == 'r':
        col_nam = ['btc_close', 'btc_volume', 'GLD_close', 'LMBA_price',
                   'GLD_market_cap', 'SP500', 'crypto_volume', 'btc_close_roc',
                   'btc_volume_roc', 'crypto_volume_roc', 'gold_close_roc',
                   'lmba_gold_roc', 'gold_volume_roc', 'SP500_roc', 'dayofweek_0',
                   'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4',
                   'dayofweek_5', 'dayofweek_6', 'year_2013', 'year_2014',
                   'year_2015', 'year_2016', 'year_2017', 'year_2018']
    else:
        col_nam = ['btc_close', 'GLD_close']

    return(data[col_nam].copy())

def get_minmax_params(rf_scores):
    """Used to obtain the parameters for refinement of RandomForest Model"""
    mnmxd = np.min(rf_scores.max_depth)
    mxmxd = np.max(rf_scores.max_depth)
    mnmxft = np.min(rf_scores.max_features)
    mxmxft = np.max(rf_scores.max_features)
    return mnmxd, mxmxd, mnmxft, mxmxft

### LSTM OBJECT ###############################################################
class PredictionModel(object):
    """And object for fitting and scoring LSTM models"""
    def __init__(self, df, scaling=False):
        self.model = None
        self.scaler = None
        new_df = df[["btc_close", "GLD_close"]].copy()

        # Take the difference so we train only on price
        # fluctuations, not raw prices.
        # df = df.diff()
        new_df.dropna(inplace=True)

        if scaling:
            scaler = preprocessing.RobustScaler()
            new_df[["btc_close",
                    "GLD_close"]] = scaler.fit_transform(new_df[["btc_close",
                                                                 "GLD_close"]])
            self.scaler = scaler

        print("Shape of df is {0}" .format(new_df.shape))
        self.df = new_df
        return

    def series_to_supervised(self, df, timesteps=1, lag=1):
        """timesteps is size of time window used for prediction
            lag is time interval after which it predicts
        """
        # Perform the reshaping for time series.
        # input sequence, count down
        col3 = list()
        for i in range(timesteps, 0, -1):
            col3.append(df.shift(i))

        # forecast sequence
        col4 = list()
        df_target = df[["btc_close"]]
        for i in range(0, lag):
            col4.append(df_target.shift(-i))

        dfx = pd.concat(col3, axis=1)
        dfy = pd.concat(col4, axis=1)

        # Drop the first few rows from BOTH which will have nans
        dfx = dfx.iloc[timesteps:]
        dfy = dfy.iloc[timesteps:]
        return dfx, dfy

    def train_lstm(self, timesteps=1, write_model=False):
        print("Training LSTM model for BTC")
        # 1. Create and compile the Model.
        lag = 1

        v1, v2 = self.series_to_supervised(self.df, timesteps=timesteps, lag=lag)
        values_X, values_Y = v1.values, v2.values

        n_features = self.df.shape[1]
        total = len(values_Y)
        train_size = int(total * 0.8)
        mult = timesteps * n_features
        train_size -= (train_size % mult)
        test_size = total - train_size
        test_size -= (test_size % mult)

        train_X = values_X[:train_size, :]
        test_X = values_X[train_size:, :]
        train_Y = values_Y[:train_size, :]
        test_Y = values_Y[train_size:, :]

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
        test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))
        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

        model = Sequential()
        neurons = 512
        model.add(LSTM(neurons, return_sequences=True,
                       input_shape=(timesteps, train_X.shape[2])))
        model.add(keras.layers.Flatten())

        output_size = 1
        model.add(Dense(units=output_size))
        model.add(Activation("linear"))
        model.compile(loss="mae", optimizer="adam")
        print(model.summary())

        x_train, x_test = train_X[
                          0:train_size, :], train_X[train_size:total, :]
        y_train = train_Y[0:train_size, :]
        # y_test = train_Y[train_size:total, :]

        # 2. Train the Model.
        history = model.fit(x_train, y_train, epochs=16, batch_size=8, \
                            verbose=2, validation_data=(test_X, test_Y))

        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        #pyplot.show()

        # 3. Save the Model.
        if write_model:
            model.save("btc_lstm_model.h5")

        self.model = model
        return test_X, test_Y

    def lstm_predict(self, test_X, test_y, timesteps):
        """ Predict future price using trained LSTM network in Keras """
        # make a prediction
        yhat = self.model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], timesteps, test_X.shape[2]))
        # drop the timesteps dimension
        test_X = test_X[:,0,:]

        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
        if self.scaler is not None:
            inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
        if self.scaler is not None:
            inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        # calculate RMSE
        rmse = math.sqrt(metrics.mean_squared_error(inv_y, inv_yhat))
        return rmse

    def load_lstm_model(self, savemodel=False):
        self.model = load_model("btc_lstm_model.h5")
        return

def timetodate(timestamp):
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')

def datetotime(date):
    return datetime.strptime(date, '%Y-%m-%d').timestamp()

### RANDOM FORREST ############################################################
def my_testTrain_split(data, feature_names, label_name):
    """create test/train split of data based on model"""
    X = data[feature_names] # features
    y = data[label_name]    # response
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# necessary definitions for scoring
rmsle_score = make_scorer(rmsle)
rmse_score = make_scorer(rmse)

def my_randomForest(X_train, y_train, mn_d=1, mx_d=100,
                    mn_ft=0.1, mx_ft=1.0, eps=10):
    """Try different model fits to obtain model params, return df of scores"""

    chpl = pd.DataFrame()
    num_epoch = eps
    n_estimators = 100
    mxd, mxft, scr = [], [], []

    for epoch in range(num_epoch):
        max_depth = int(np.random.uniform(mn_d, mx_d))
        max_features = np.random.uniform(mn_ft, mx_ft)

        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      n_jobs=-1,
                                      random_state=37)
        score = cross_val_score(model,
                                X_train, y_train,
                                cv=20, scoring=rmsle_score).mean()

        print(
            "max_depth = {0}, max_features = {1:.6f}, score = {2:.5f}".format(
                max_depth,
                max_features,
                score))
        mxd.append(max_depth)
        mxft.append(max_features)
        scr.append(score)

    chpl = chpl.assign(
        **{'max_depth': mxd, 'max_features': mxft, 'score': scr})
    chpl.sort_values("score", ascending=True, inplace=True)
    return(chpl)

def fit_RF(n_estimators, max_depth, max_features, X, y):
    """Fit RF to X, y, return fitted RandomForrest Model, return rf object"""
    rfm = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                max_features=max_features,
                                n_jobs=-1,
                                random_state=23)
    rfm.fit(X, y)
    return rfm

### LSTM MODEL RUN ############################################################
def run_lstm(data):
    timesteps = 16
    pred = PredictionModel(data, scaling=True)
    x_test, y_test = pred.train_lstm(timesteps=timesteps, write_model=True)
    rmse = pred.lstm_predict(x_test, y_test, timesteps=timesteps)
    print('RMSE of LSTM predictions: %.3f' % rmse)
    return pred

### START TABLE-SETTINGS ######################################################
# rmsle_score = make_scorer(rmsle)
# rmse_score = make_scorer(rmse)
# core_data = get_combined_data()
#
# ### END TABLE-SETTINGS #######################################################
#
# ### Random Forest Model I ####################################################
# data_r = prep_data(core_data, 'r')
# label_name, feature_names = label_and_feat(data_r, 'r')
# X_train, X_test, y_train, y_test = my_testTrain_split(data_r, feature_names,
#                                                       label_name)
#
# fst_pass = my_randomForest(X_train, y_train) # 10 epochs is default
# mnmxd, mxmxd, mnmxft, mxmxft = get_minmax_params(fst_pass) # refined params
# refined = my_randomForest(X_train, y_train,
#                            mnmxd, mxmxd, mnmxft, mxmxft, 5)
# refined = refined.iloc[refined.score.idxmin(), :] #obtain min score
# rfm = fit_RF(3000,refined.max_depth,refined.max_features, X_train, y_train)
# scores = my_crossval(rfm, X_train, y_train, "r-forrest")
#
# rfm_preds = rfm.predict(X_test)
# rfm_preds[:10]
# y_test[:10].values
# act_v_predict_plot(rfm_preds, y_test, 60)
#
# ### Random Forest Model II ####################################################
# data_j = prep_data(core_data, 'j')
# label_name, feature_names = label_and_feat(data_j, 'j')
# X_train, X_test, y_train, y_test = my_testTrain_split(data_j, feature_names,
#                                                       label_name)
#
# fst_pass = my_randomForest(X_train, y_train) # 10 epochs is default
# mnmxd, mxmxd, mnmxft, mxmxft = get_minmax_params(fst_pass) # refined params
# refined = my_randomForest(X_train, y_train,
#                            mnmxd, mxmxd, mnmxft, mxmxft, 5)
# refined = refined.iloc[refined.score.idxmin(), :] #obtain min score
# rfm = fit_RF(3000,refined.max_depth,refined.max_features, X_train, y_train)
# scores = my_crossval(rfm, X_train, y_train, "j-forrest")
#
# rfm_preds = rfm.predict(X_test)
# rfm_preds[:10]
# y_test[:10].values
# act_v_predict_plot(rfm_preds, y_test, 60)
#
# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: round(x, 4), rfm.feature_importances_),
#                  feature_names), reverse=True))
#
# ### LSTM Model ################################################################
# data = prep_data(core_data, 'v')
# def run_lstm(data):
#     timesteps = 16
#     pred = PredictionModel(data, scaling=True)
#     x_test, y_test = pred.train_lstm(timesteps=timesteps, write_model=True)
#     rmse = pred.lstm_predict(x_test, y_test, timesteps=timesteps)
#     print('RMSE of LSTM predictions: %.3f' % rmse)
#     return pred
#
# data = prep_data(core_data, 'v')
# lstm = run_lstm(data)