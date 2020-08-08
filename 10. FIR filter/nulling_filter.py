import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

fs = 100
t = np.linspace( 0, 1, fs, endpoint = False )	# 定義時間陣列
x1 = np.cos( 2 * np.pi * 10 * t )				# 原始訊號
x2 = np.cos( 2 * np.pi * 20 * t )				# 干擾訊號
x = x1 + x2

b = np.array( [ 1, -2 * np.cos( 2 * np.pi * 20 / fs ), 1 ] )	
y = signal.lfilter( b, 1, x )					# FIR濾波

plt.figure( 1 )									# 繪圖
plt.plot( t, x )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -2, 2 ] )

plt.figure( 2 )									
plt.plot( t, x1, '--', t, y, '-' )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -2, 2 ] )

plt.show( )