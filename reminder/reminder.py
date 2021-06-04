import numpy as np  # pip install numpy
import simpleaudio as sa # pip install 
import time

frequency = 440         # frequency note 440 Hz
fs = 44100              # samples per second
seconds = 0.2           # Note duration of 3 seconds
reminder_seconds = 5    # beeping every x seconds
t = np.linspace(0, seconds, seconds * fs, False) # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
note = np.sin(frequency * t * 2 * np.pi) # Generate a sine wave of given frequency
audio = note * (2**15 - 1) / np.max(np.abs(note)) # Ensure that highest value is in 16-bit range
audio = audio.astype(np.int16) # Convert to 16-bit data

while True:
    for i in range(3):
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        time.sleep(seconds + 0.1)
    time.sleep(reminder_seconds)
