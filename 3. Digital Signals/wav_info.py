import wave

filename = input( "Please enter file name: " )
wav = wave.open( filename, 'rb' )

num_channels = wav.getnchannels( )	# 通道數
sampwidth	= wav.getsampwidth( )	# 樣本寬度
frame_rate	= wav.getframerate( )	# 取樣率
num_frames	= wav.getnframes( )		# 音框數
comptype	= wav.getcomptype( )	# 壓縮型態
compname	= wav.getcompname( )	# 壓縮名稱

print( "Number of Channels =", num_channels )
print( "Sample Width =", sampwidth )
print( "Sampling Rate =", frame_rate )
print( "Number of Frames =", num_frames )
print( "Comptype =", comptype )
print( "Compname =", compname )

wav.close( )