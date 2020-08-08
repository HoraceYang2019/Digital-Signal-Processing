import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000 )			# 定義時間陣列
x = signal.square( 2 * np.pi * 5 * t )	# 產生方波

plt.plot( t, x )						# 繪圖
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -1.2, 1.2 ] )

plt.show( )