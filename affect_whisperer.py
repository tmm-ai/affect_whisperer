import time
import torch
import torch.nn as nn
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForSequenceClassification
import matplotlib

def record_audio(duration, threshold):
    """
    Records audio for a specified duration (in seconds) while the average absolute amplitude is above the threshold.
    Returns the recorded audio as a NumPy array or None if no frames were recorded.
    """
    # Constants for the recording setup
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Initialize the PyAudio library
    p = pyaudio.PyAudio()

    # Set up the recording stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Initialize an empty list to store the recorded frames
    frames = []

    # Record audio for the specified duration (in seconds) while the average absolute amplitude is above the threshold
    for i in range(0, int(RATE / CHUNK * duration)):
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        if np.abs(data).mean() > threshold:
            frames.append(data)
        if i % (RATE // CHUNK) == 0:  # Print frames count every second
            print(f"Recorded {len(frames)} frames so far.")

    # Stop the recording stream and close the PyAudio library
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Return the concatenated frames as a NumPy array or None if no frames were recorded
    if not frames:
        return None
    return np.concatenate(frames, axis=0)

def predict_emotion(audio, device, model):
    """
    Takes the recorded audio, the device (CPU or GPU) to perform the computations, and a pre-trained model.
    Returns the probabilities for each emotion class as a NumPy array.
    """
    # Convert the audio data to a PyTorch tensor and send it to the specified device
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    # Pass the audio data through the model and get the logits (raw output values)
    logits = model(audio).logits.squeeze(0)

    # Compute the probabilities of each emotion using the softmax function
    emotion_probs = nn.Softmax(dim=0)(logits).detach().cpu().numpy()

    # Return the emotion probabilities as a NumPy array
    return emotion_probs

def get_emotions():
    """
    Returns a tuple containing the list of emotions and a dictionary of their corresponding colors.
    """
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
    return EMOTIONS, EMOTION_COLORS

def bar_chart(emotion_probs):
    """
    Updates the bar chart with the latest emotion probabilities.
    """
    EMOTIONS, EMOTION_COLORS = get_emotions()
    print(emotion_probs)
    plt.clf()
    plt.bar(range(8), emotion_probs, color=[EMOTION_COLORS[e] for e in EMOTIONS])
    plt.xticks(range(8), EMOTIONS, rotation=45)
    plt.ylim(0, 1)
    plt.title("Emotion Prediction")
    plt.xlabel("Emotion")
    plt.ylabel("Probability")
    plt.pause(0.1)

def line_chart(emotion_history, duration):
    """
    Plots a line chart of emotions over time based on the provided emotion history.
    """
    # Convert the emotion history to a NumPy array
    plt.ioff()
    emotion_history = np.array(emotion_history)

    # Clear the current figure
    plt.clf()

    # Get the emotions and their corresponding colors
    EMOTIONS, EMOTION_COLORS = get_emotions()

    # Plot the emotion probabilities over time for each emotion
    for i in range(8):
        plt.plot(emotion_history[:, i], label=EMOTIONS[i], color=EMOTION_COLORS[EMOTIONS[i]])

    # Adjust x-axis based on the amount of data recorded
    elapsed_time = emotion_history.shape[0]
    xticks = np.arange(0, elapsed_time, duration)
    plt.xticks(xticks, xticks * duration)  # Assuming the time interval is 2 seconds

    # Set the title, labels, and legend for the line chart
    plt.title("Emotion Change Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability of Emotion")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust the layout of the plot and display it
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the program. It sets up the device, loads the pre-trained model, and records and analyzes emotions.
    """
    # Set the backend for matplotlib
    matplotlib.use('TkAgg')

    # Set the device for running the model (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    model_path = "/Users/Tom/Dropbox/PycharmProject/Interview_buddy/wav2vec2_emotion_model_Ravdess1e5_30epoch_8160"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Record and analyze emotions
    print("Please answer the following question:")
    print("Tell me about a time when you faced a difficult challenge and how you overcame it.")
    time.sleep(1)
    elapsed_time = 0
    duration = 2
    threshold = 80
    emotion_history = []

    # Record audio and predict emotions until the audio is empty (no audio recorded)
    while True:
        plt.ion()
        audio = record_audio(duration, threshold)
        if audio is None:
            line_chart(emotion_history, duration)
        elapsed_time += duration
        if len(audio) > 1:
            emotion_probs = predict_emotion(audio, device, model)
            emotion_history.append(emotion_probs)
            # Update the bar chart with the latest emotion probabilities
            print(emotion_probs)
            bar_chart(emotion_probs)
        else:
            break


if __name__ == "__main__":
    main()

