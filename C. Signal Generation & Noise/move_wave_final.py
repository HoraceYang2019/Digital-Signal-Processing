import time
import pyaudio
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from queue import Queue

queue = Queue(maxsize=300000)
queue2 = Queue(maxsize=300000)

p = pyaudio.PyAudio()
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 5


stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
play_stream = p.open(format=sample_format, channels=channels, rate=fs, output=True)

def Wavedata():
    print("開始")
    start_time = time.time()
    while (time.time() - start_time) < seconds:
        data = stream.read(1025)
        audio_data = np.frombuffer(data, dtype=np.int16)
        queue.put(audio_data)
        queue2.put(data)
    print("結束")
    stream.stop_stream()
    stream.close()
    p.terminate()
    

def Play_wave():
    start_time = time.time()
    while (time.time() - start_time) < seconds:
        queuedata = queue2.get()
        queue2.task_done()
        play_stream.write(queuedata)
    play_stream.stop_stream()
    play_stream.close()
    p.terminate()

    


Wavedata_thread1 = threading.Thread(target=Wavedata)
Play_wave_thread2 = threading.Thread(target=Play_wave)
Wavedata_thread1.start()
Play_wave_thread2.start()


plt.ion()
fig1 = plt.figure(1)
Amplitude_data = [0]
while True:
    for i in range(10):
        queuedata = queue.get()
        queue.task_done()
        for i in queuedata:
            Amplitude_data.append(i)
    if len(Amplitude_data) >= 88200:
        Amplitude_data = Amplitude_data[1024:]
    plt.plot(Amplitude_data,'-r')
    plt.pause(0.01)
    fig1.clf()
    if queue.empty():
        print("stop")
        break




