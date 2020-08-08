import numpy as np
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000, endpoint = False )	# 定義時間陣列

x = np.cos( 2 * np.pi * 5 * t )					# 產生弦波
a = np.linspace( 1, 0, 1000, endpoint = False ) # 產生淡出陣列 
x = x * a                                       # 套用淡出效果 

plt.plot( t, x )								# 繪圖
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -1.2, 1.2 ] )

plt.show( )