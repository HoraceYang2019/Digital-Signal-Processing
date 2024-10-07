import pandas as pd 
import datetime
from matplotlib import pyplot
import pandas as pd
from pmdarima.arima import ndiffs

def parser(x):
    return datetime.datetime.strptime(x, '%m.%d.%Y %h:%m:%s') # 01.01.2009 00:10:00

#df = pd.read_csv('./datasets\\jena_climate_2009_2016.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
df = pd.read_csv('./datasets\\jena_climate_2009_2016.csv', header=0, parse_dates=[0], index_col=0)

print(df.head())
df['T (degC)'].plot()
#plt.rcParams["figure.figsize"] = (20,5)
pyplot.show()

# 算出推薦的差分次數
d =  ndiffs(df['T (degC)'],  test="adf")
print(d) # 1

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df['T (degC)'], lags = 15, method = "ols")

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['T (degC)'], lags = 150) 