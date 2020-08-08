import numpy as np
import matplotlib.pyplot as plt

n = np.array( [ 0, 1, 2, 3, 4, 5 ] )	# 定義n陣列
x = np.array( [ 1, 2, 4, 3, 2, 1 ] )	# 定義x陣列

plt.stem( n, x )						# 繪圖
plt.xlabel( 'n' )
plt.ylabel( 'x[n]' )
plt.show( )