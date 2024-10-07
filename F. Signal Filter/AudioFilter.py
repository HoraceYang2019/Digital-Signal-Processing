# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:26:47 2023

@author: hao
"""
import numpy as np
import wave
import struct
import scipy.signal as signal

original_file = "Waves\exp_sin.wav"	# 
filtered_file = "Waves\exp_sin_fd.wav"

amplitude = 50000           # 
frequency = 256			# 
duration = 3				# in sec
fs = 44100				   	#
num_samples = duration * fs
 
num_channels = 1			# 
sampwidth = 2				# 
num_frames = num_samples	# 
comptype = "NONE"		   	# 
compname = "not compressed" # 

# In[]
t = np.linspace( 0, duration, fs, endpoint = False )
x = amplitude * np.exp( -t ) * np.sin( 2 * np.pi * frequency * t )
#x = amplitude * np.cos( 2 * np.pi * frequency * t )

x = np.pad( x, ( 0, 4 * fs ), 'constant' )
y = np.clip( x, -30000, 30000 )

wav_file = wave.open(original_file, 'wb' )
wav_file.setparams(( num_channels, sampwidth, fs, num_frames, comptype, compname )) 

for s in y :
   wav_file.writeframes( struct.pack( 'h', int ( s ) ) )

wav_file.close( ) 

# In[]
b = np.array( [ 1 ] )
a = np.zeros( duration * fs )

num_echos = 6
for i in range( num_echos ):
	a[int(i * fs/4)] = 1 - i *0.2 

# In[]
y = signal.lfilter( x, b, a )
y = np.clip( y, -30000, 30000 )

wav_file = wave.open( filtered_file, 'w' )
wav_file.setparams(( num_channels, sampwidth, fs, num_frames, comptype, compname )) 

for s in y :
   wav_file.writeframes( struct.pack( 'h', int ( s ) ) )

wav_file.close( )
