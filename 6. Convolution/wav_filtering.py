import numpy as np
import wave
from scipy.io.wavfile import read, write
import struct
import scipy.signal as signal

def average_filtering( x, filter_size ):
	h = np.ones( filter_size ) / filter_size
	y = np.convolve( x, h, 'same' )
	return y
	
def gaussian_filtering( x, sigma):
	filter_size = 6 * sigma + 1  
	gauss = signal.gaussian( filter_size, sigma )	
	sum = np.sum( gauss )                   		
	gauss = gauss / sum
	y = np.convolve( x, gauss, 'same' )     
	return y
	
def normalization( x, maximum ):
	x_abs = abs( x )
	max_value = max( x_abs )
	y = x / max_value * maximum
	return y
	
def main( ):
	infile  = input( "Input File: " )	
	outfile = input( "Output File: " )
	
	# ----------------------------------------------------
	#  輸入模組
	# ----------------------------------------------------	
	wav = wave.open( infile, 'rb' )
	num_channels = wav.getnchannels( )	# 通道數
	sampwidth	 = wav.getsampwidth( )	# 樣本寬度
	fs			 = wav.getframerate( )	# 取樣頻率(Hz)
	num_frames	 = wav.getnframes( )	# 音框數 = 樣本數
	comptype	 = wav.getcomptype( )	# 壓縮型態
	compname	 = wav.getcompname( )	# 無壓縮
	wav.close( )

	sampling_rate, x = read( infile )	# 輸入訊號

	# ----------------------------------------------------
	#  DSP 模組
	# ----------------------------------------------------	
	print( "Filtering" )
	print( "(1) Average Filtering" )
	print( "(2) Gaussian Filtering" )
	
	choice = eval( input( "Please enter your choice: " ) )
	
	if choice == 1:
		filter_size = eval( input( "Filter Size = " ) )
		y = average_filtering( x, filter_size )
	elif choice == 2:
		sigma = eval( input( "Sigma = " ) )
		y = gaussian_filtering( x, sigma )
	else:
		print( "Your choice is not supported!" )
		y = x
	
	y = normalization( x, 30000 )
	
	# ----------------------------------------------------
	#  輸出模組
	# ----------------------------------------------------		
	wav_file = wave.open( outfile, 'w' )
	wav_file.setparams(( num_channels, sampwidth, fs, num_frames, comptype, compname )) 

	for s in y:
		wav_file.writeframes( struct.pack( 'h', int ( s ) ) )

	wav_file.close( ) 
	
main( )