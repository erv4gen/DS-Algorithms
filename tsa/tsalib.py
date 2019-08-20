import os
import sys , time , traceback , itertools , warnings , ipdb , uuid
import datetime as dt

from tqdm import tqdm

import pandas as pd
import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')

adj_close = 'adj_close'
def get_market_ts(date,ticker,data_source):
    if data_source=='web':
        start = '2000-01-01'
        end = date

        get_px = lambda x: web.DataReader(x, 'yahoo', start=start, end=end)

        # symbols = ['SPY','TLT','MSFT']
        # # raw adjusted close prices
        # data = pd.DataFrame({sym:get_px(sym)['Adj Close'] for sym in symbols})

        ticker_data = get_px(ticker)
        
        ticker_data = ticker_data.dropna()
        ticker_data = ticker_data.rename({'Adj Close':adj_close},axis=1)
        ticker_data.columns = [x.lower() for x in ticker_data.columns]
        
        ticker_data['volatility'] = calc_volatility(ticker_data)

        ticker_data['return'] = 100* ticker_data[adj_close].pct_change()

    else:
        pass
            #ticker_data = get_px(ticker)
        # log returns
        #lrets = np.log(ticker_data['Adj Close']/ticker_data['Adj Close'].shift(1)).dropna()
        
    return ticker_data

def calc_volatility(ticker_data):
    '''This volatility is different from VIX volatility and track only the price candle size'''
    
    return 100*(ticker_data.high - ticker_data.low)/ticker_data[adj_close]


def tsplot(y, lags=None, figsize=(8, 6), style='ggplot',title='Time Series Analysis Plots'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return