import pandas as pd
from scipy.signal import argrelextrema
import pandas_ta as ta
import numpy as np


from scipy import stats

df = pd.read_csv("eurusd_1h_ohlc.csv")
df.columns= ['time','open','high','low','close','volume']



peaks = argrelextrema(df['close'].values, np.greater,order=15)[0]
df['peaks'] = df['close'][peaks]
df.peaks.fillna(method="ffill",inplace=True)

depths = argrelextrema(df['close'].values, np.less,order=15)[0]
df['depth'] = df['close'][depths]
df.depth.fillna(method="ffill",inplace=True)


df['sma_10'] = ta.sma(df['close'],length=20)
df['sma_30'] = ta.sma(df['close'],length=50)
df['sma_100'] = ta.sma(df['close'],length=100)
df['sma_200'] = ta.sma(df['close'],length=200)

df['ema_10'] = ta.ema(df['close'],length=20)
df['ema_30'] = ta.ema(df['close'],length=50)
df['ema_100'] = ta.ema(df['close'],length=100)
df['ema_200'] = ta.ema(df['close'],length=200)

df['cci_20'] = ta.momentum.cci(df['high'],df['low'],df['close'],length=20)
df['rsi'] = ta.rsi(df['close'],length=10)

rsi_peaks = argrelextrema(df['rsi'].values, np.greater,order=15)[0]
df['rsi_peaks'] =  df['rsi'][rsi_peaks]
df.rsi_peaks.fillna(method="ffill",inplace=True)

rsi_depths = argrelextrema(df['rsi'].values, np.less,order=15)[0]
df['rsi_depth'] = df['rsi'][rsi_depths]
df.rsi_depth.fillna(method="ffill",inplace=True)
#macd
macd = ta.macd(df['close'])
df['macd'] = macd['MACDh_12_26_9']
df['histogram']= macd['MACDh_12_26_9']
df['signal'] = macd['MACDs_12_26_9']

stoch = ta.stoch(df['high'],df['low'],df['close'])
df['STOCHk_14_3_3'] = stoch['STOCHk_14_3_3']
df['STOCHd_14_3_3'] = stoch['STOCHd_14_3_3']

ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
#For the visible period: spanA, spanB, tenkan_sen, kijun_sen, and chikou_span columns
df['spanA'] = ichimoku[0]['ISA_9']
df['spanB'] = ichimoku[0]['ISB_26']
df['tenkan'] = ichimoku[0]['ITS_9']
df['kejan'] = ichimoku[0]['IKS_26']
df['chikouSpan'] = ichimoku[0]['ICS_26']
#For the forward looking period: spanA and spanB columns
spanA = ichimoku[1]['ISA_9']
spanB = ichimoku[1]['ISB_26']

#standardDeviation
df['stdev'] = ta.stdev(df['close'],length=30)

#cci
df['cci'] = ta.cci(df['high'], df['low'], df['close'])
#adx
adx = ta.adx(df['high'], df['low'], df['close'])
df['adx_14'] = adx['ADX_14']
df['DMP_14'] = adx['DMP_14']
df['DMN_14'] = adx['DMN_14']

#log_return 
df['log_return'] = ta.log_return(df['close'])
#linreg
df['linreg']=  ta.linreg(df['close'])

#psar
psar = ta.psar(df['high'],df['low'],df['close'])
df['psarl'] = psar['PSARl_0.02_0.2']
df['psars'] = psar['PSARs_0.02_0.2']
df['psaraf'] = psar['PSARaf_0.02_0.2']
df['psarr'] = psar['PSARr_0.02_0.2']
#zscore
df['zscore'] = ta.zscore(df['close'])

#slope
df['slope'] = ta.slope(df['close'])

#squeeze
squeeze = ta.squeeze(df['high'],df['low'],df['close'])
df['SQZ_20_2.0_20_1.5'] = squeeze['SQZ_20_2.0_20_1.5']
df['SQZ_ON'] = squeeze['SQZ_ON']
df['SQZ_OFF'] = squeeze['SQZ_OFF']
df['SQZ_NO'] = squeeze['SQZ_NO']


#variance
df['variance'] = ta.variance(df['close'])

supertrend = ta.supertrend(df['high'], df['low'], df['close'])
df['SUPERT_7_3.0'] = supertrend['SUPERT_7_3.0']
df['SUPERTd_7_3.0'] = supertrend['SUPERTd_7_3.0']
df['SUPERTl_7_3.0'] = supertrend['SUPERTl_7_3.0']
df['SUPERTl_7_3.0'] = supertrend['SUPERTs_7_3.0']


#bbands
bands = ta.bbands(df['close'])
df['loweband'] = bands['BBL_5_2.0']
df['middleband'] = bands['BBM_5_2.0']
df['upperband'] = bands['BBU_5_2.0']
df['BBB_5_2.0'] = bands['BBB_5_2.0']
df['BBP_5_2.0'] = bands['BBP_5_2.0']

#wma
df['wma'] = ta.wma(df['close'])

#vwma
df['vwma'] = ta.vwma(df['close'],df['volume'])
#increasing
df['increasing'] = ta.increasing(df['close'])

aroon = ta.aroon(df['high'], df['low'])
df['AROOND_14'] = aroon['AROOND_14']
df['AROONU_14'] = aroon['AROONU_14']
df['AROONOSC_14'] = aroon['AROONOSC_14']


#ATR should be added
df['atr'] = ta.atr(df['high'], df['low'], df['close'])
#variance
df['variance'] = ta.variance(df['close'])
#df.dropna(inplace = True)
df = df[200:]
print(df.shape)

pattern = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'])


df = pd.concat([df,pattern],axis=1)
df.index = df['time']
df.drop(['time'],inplace = True,axis=1)

#drop the columns which are all na 
df.dropna(inplace = True,axis=1,how='all')
#df.fillna(-9999999,inplace=True)
#df.isna().sum()
#df.info()



# Exponential Smoothing
alpha = 0.2  # Smoothing factor
df['Exp Smoothing'] = df['close'].ewm(alpha=alpha, adjust=False).mean()

# Log Returns
#data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Volatility (Standard Deviation)
#volatility = np.std(data['Log Returns'])

# Correlation
df['correlation'] = df['close'].corr(df['volume'])

# Linear Regression
#slope, intercept, r_value, p_value, std_err = stats.linregress(df['open'], df['close'])

# Fourier Transform
#df['fourier_transform'] = np.fft.fft(df['close'])
#df['frequencies'] = np.fft.fftfreq(len(df), d=1)  # Assuming data is daily


#scaling
df.replace([np.nan,np.inf, -np.inf], -99999, inplace=True)
from sklearn import preprocessing
import numpy as np

#df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

#standardize the data, column wise it scales based on a mean of 0 and std of 1
#df2 = preprocessing.scale(df)


#scaling in a way that sum of each row becomes one
df2=preprocessing.normalize(df,norm='l1')
df2 = pd.DataFrame(df2, columns=df.columns)
df2.index=df.index
print(df)

"""
#scaling between 0 and 1, min value is 0 and max is 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on your data and transform it
df_scaled = scaler.fit_transform(df)
# Create a new DataFrame with the scaled data
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
# Now your data is scaled between 0 and 1
"""
df2.to_csv('featured_data.csv')


print(df2.shape)


"""
# Black-Scholes Formula (for educational purposes, not complete implementation)
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

# Example usage of Black-Scholes formula
option_price = black_scholes(S=100, K=95, T=0.5, r=0.05, sigma=0.2)
"""
# These examples cover a subset of the concepts mentioned earlier.
# You can integrate these calculations into your financial analysis pipeline.
