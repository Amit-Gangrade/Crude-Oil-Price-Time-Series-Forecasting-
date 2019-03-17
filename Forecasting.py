import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6 
import matplotlib

df = pd.read_csv("File 1.csv")

#indexedDataset = df
#indexedDataset = indexedDataset.set_index('Date')
#indexedDataset.index

df['Date'] = pd.to_datetime(df['Date'])
indexedDataset = df.set_index(['Date'])

from datetime import datetime
indexedDataset.head(5)

plt.plot(indexedDataset)

#Determine rolling statistics 

rolmean = indexedDataset.rolling(window = 12).mean()
rolstd =  indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

#Plot rolling statistics

orig = plt.plot(indexedDataset, color='blue', label='origional')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color = 'black', label = 'Rolling Strd')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.show(block=False)

# Trandform it to the log of the dataset 

indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)

movingaverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingaverage,color='red')

datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingaverage
datasetLogScaleMinusMovingAverage.head(12)

# Remove NAN Values 
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

# Determine again the rolling test\
def test_stationarity(timeseries):
    
   # Determine the rolling statistics 
    
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
   # plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Origional')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD,color='black', label='Rolling STD')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standerd Deviation')
    plt.show(block=False)
    
 
        
    test_stationarity(datasetLogScaleMinusMovingAverage)    
    
    exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
    plt.plot(indexedDataset_logScale)
    plt.plot(exponentialDecayWeightedAverage, color='red')

    datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
    test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)
   
    datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
    plt.plot(datasetLogDiffShifting)
   
    datasetLogDiffShifting.dropna(inplace=True)
    test_stationarity(datasetLogDiffShifting)
   
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(indexedDataset_logScale)
   
    trend = decomposition.trend
    seasonal = decomposition.seasonal 
    residual = decomposition.resid
   
    plt.subplot(411)
    plt.plot(indexedDataset_logScale, label='Original')
    plt.legend(loc = 'best')

    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc = 'best')        

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc = 'best')         
   
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc = 'best')         
    plt.tight_layout()
   
    decomposedLogData = residual 
    decomposedLogData.dropna(inplace=True)
    test_stationarity(decomposedLogData)  
   
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)

lag_pacf = pacf(datasetLogDiffShifting, nlags = 20, method = 'ols')

# plot ACF:

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle = '--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')   
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')  
plt.title('Autocorrelation Function') 
   
#plot PACF

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle = '--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')   
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')  
plt.title('Partial Autocorrelation Function') 

plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

# AR Model 

model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting["Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"])**2))
print('Plotting AR Model')

# MA Model 
    
model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"])**2))
print('Plotting AR Model')

# ARIMA Model 
    
model = ARIMA(indexedDataset_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"])**2))
print('Plotting ARIMA Model')

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

# Converted to calculative sum 

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['Cushing, OK WTI Spot Price FOB (Dollars per Barrel)'].ix[0],index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value = 0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)

indexedDataset_logScale 

results_ARIMA.forecast(steps=6)





   
        
    
    
    
   

