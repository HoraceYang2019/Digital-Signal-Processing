import numpy as np
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000, endpoint = False )	# 定義時間陣列

f1 = 2 											# 定義基礎頻率
x1 = np.cos( 2 * np.pi * f1 * t )	  			# 產生第1個弦波
x2 = np.cos( 2 * np.pi * 2 * f1 * t )			# 產生第2個弦波
x = x1 + x2										# 產生諧波

plt.figure( 1 )									# 繪圖
plt.plot( t, x1 )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -2, 2 ] )

plt.figure( 2 )
plt.plot( t, x2 )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -2, 2 ] )

plt.figure( 3 )
plt.plot( t, x )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -2, 2 ] )

plt.show( )