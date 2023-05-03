import time
import torch
import torch.nn as nn
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForSequenceClassification
import matplotlib
matplotlib.use('TkAgg') # pychar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/Users/Tom/Dropbox/PycharmProject/Interview_buddy/wav2vec2_emotion_model_Ravdess1e5_30epoch_8160"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)

# config = AutoConfig.from_pretrained(model_dir)
model.eval()

# Define the recording function
def record_audio(duration, threshold):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        if np.abs(data).mean() > threshold:
            frames.append(data)
        if i % (RATE // CHUNK) == 0:  # Print frames count every second
            print(f"Recorded {len(frames)} frames so far.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    if not frames:
        return None
    return np.concatenate(frames, axis=0)


# Define the emotion prediction function
def predict_emotion(audio):
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(audio).logits.squeeze(0)
    emotion_probs = nn.Softmax(dim=0)(logits).detach().cpu().numpy()
    return emotion_probs



# Record and analyze emotions
print("Please answer the following question:")
print("Tell me about a time when you faced a difficult challenge and how you overcame it.")
time.sleep(1)
elapsed_time = 0
duration = 2
threshold = 80
emotion_history = []
plt.ion()
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_COLORS = {
    "neutral": "gray",
    "calm": "blue",
    "happy": "yellow",
    "sad": "purple",
    "angry": "red",
    "fearful": "orange",
    "disgust": "green",
    "surprised": "pink",
}

while True:
    audio = record_audio(duration, threshold)
    if audio is None:
        break
    elapsed_time += duration
    if len(audio) > 1:
        emotion_probs = predict_emotion(audio)
        emotion_history.append(emotion_probs)
        # Update the bar chart
        print(emotion_probs)
        plt.clf()
        plt.bar(range(8), emotion_probs, color=[EMOTION_COLORS[e] for e in EMOTIONS])
        plt.xticks(range(8), EMOTIONS, rotation=45)
        plt.ylim(0, 1)
        plt.title("Emotion Prediction")
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.pause(0.1)
    else:
        break

plt.ioff()
# Plot the line chart of emotions over time

emotion_history = np.array(emotion_history)
plt.clf()
for i in range(8):
    plt.plot(emotion_history[:, i], label=EMOTIONS[i], color=EMOTION_COLORS[EMOTIONS[i]])

# Adjust x-axis based on the amount of data recorded
elapsed_time = emotion_history.shape[0]
xticks = np.arange(0, elapsed_time, duration)
plt.xticks(xticks, xticks * duration)  # Assuming the time interval is 5 seconds

plt.title("Emotion Change Over Time")
plt.xlabel("Time (2 seconds interval)")
plt.ylabel("Probability")

# Place the legend outside the graph on the right side
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()