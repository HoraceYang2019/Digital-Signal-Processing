import numpy as np
import matplotlib.pyplot as plt

t = np.linspace( 0, 0.1, 1000, endpoint = False )	# 定義時間陣列	

f1 = 20												# 低頻頻率
f2 = 200											# 高頻頻率
x = np.cos( 2 * np.pi * f1 * t ) * np.cos( 2 * np.pi * f2 * t )
envelop1 =  np.cos( 2 * np.pi * f1 * t )			# 包絡
envelop2 = -np.cos( 2 * np.pi * f1 * t )

plt.plot( t, x, '-' )								# 繪圖
plt.plot( t, envelop1, '--', color = 'b' )
plt.plot( t, envelop2, '--', color = 'b' )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 0.1, -1, 1 ] )

plt.show( )