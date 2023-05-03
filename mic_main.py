import sys
import wave
# Import the pyaudio module  # Add the path to the pyaudio module to the Python path
# sys.path.append('/Users/Tom/opt/anaconda3/lib/python3./site-packages')
import pyaudio
from API_keys import luxand_API, empath_API, openai_API
# Set up the PyAudio object
p = pyaudio.PyAudio()
import speech_recognition as sr
# Open a stream using default input device
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Record audio for 5 seconds
frames = []
for i in range(0, int(44100 / 1024 * 4)):
    data = stream.read(1024)
    frames.append(data)

# Close the stream and PyAudio object
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
wavefile = wave.open("recorded_vs_studio.wav", "wb")
wavefile.setnchannels(1)
wavefile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wavefile.setframerate(44100)
wavefile.writeframes(b''.join(frames))
wavefile.close()
speech_recognizer = sr.Recognizer()
