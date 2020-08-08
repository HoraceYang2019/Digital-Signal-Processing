import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000, endpoint = False )	# 定義時間陣列
x = signal.chirp( t, 0, 5, 5, 'linear' )		# 產生啁啾訊號

plt.plot( t, x )								# 繪圖
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 5, -1.5, 1.5 ] )

plt.show( )