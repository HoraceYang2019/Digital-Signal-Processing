import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import threading
import wave


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()


frames = []


x = np.arange(0, 2 * CHUNK, 2)
fig, ax = plt.subplots()
ax.set_ylim(-32768, 32768)
line, = ax.plot(x, np.zeros(CHUNK), '-', lw=2) 


def audio_stream():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        )

    print("Recording...")

    while True:
        data = stream.read(CHUNK) 
        audio_data = np.frombuffer(data, dtype=np.int16) 
        line.set_ydata(audio_data) 
        fig.canvas.draw()
        fig.canvas.flush_events() 
        frames.append(data) 


audio_thread = threading.Thread(target=audio_stream)


audio_thread.start()


try:
    while True:
        plt.pause(1) 
except KeyboardInterrupt:
    print("Finished recording!")


    audio_thread.join() 
    audio.terminate() 

    # with wave.open("output.wav", "wb") as wf:
    #     wf.setnchannels(CHANNELS)
    #     wf.setsampwidth(audio.get_sample_size(FORMAT))
    #     wf.setframerate(RATE)
    #     wf.writeframes(b''.join(frames))