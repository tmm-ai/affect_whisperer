# Affect Whisperer
Affect Whisperer helps users visualize the emotions they convey in real time. Use cases range from gaining insights on emotions during meetings, interview practice, discussing sensitive matters, or simply obtaining a readout of emotions while working.

## Introduction
Currently, Affect Whisperer interprets emotions through vocal intonations. In the near future, the program will also include facial expression recognition and sentiment analysis, each with their own graphical outputs.

For vocal intonation training, the TESS and RAVDESS datasets were used. Wav2Vec2 is a deep learning model designed for automatic speech recognition. Wav2Vec stands for "Waveform to Vector," highlighting its purpose of transforming audio data into a more structured format. This model was employed to train a classifier on these datasets to identify emotions.

The Wav2Vec_intv_helper.ipynb file cleans and merges the datasets, then trains the model.

affect_whisperer.py is the ready-to-go file that uses the trained model and outputs bar charts that update every 2 seconds with new emotion levels. After the user stops talking, a line chart illustrating emotion levels throughout the entire conversation is produced.

## Dependencies - affect_whisperer.py
Python 3.x
torch
pyaudio
numpy
matplotlib
transformers


## Usage
Ensure that you have all the necessary dependencies installed.
Place the pre-trained Wav2Vec2 model in the specified path (https://drive.google.com/file/d/1-B98oQYZVgPJRVgpCmdHPLFzoTlBa3Pd/view?usp=share_link).
Run the program using python affect_whisperer.py in your terminal or command prompt.
Answer the prompted question, and the program will record and analyze your speech.
The program will display emotion probabilities in real-time using a bar chart.
Once you've finished speaking, the program will display a line chart showing how emotions changed over time.


## Functions

record_audio(duration, threshold): Records audio for a specified duration (in seconds) while the average absolute amplitude is above the threshold. Returns the recorded audio as a NumPy array or None if no frames were recorded.
predict_emotion(audio, device, model): Takes the recorded audio, the device (CPU or GPU) to perform the computations, and a pre-trained model. Returns the probabilities for each emotion class as a NumPy array.
get_emotions(): Returns a tuple containing the list of emotions and a dictionary of their corresponding colors.
bar_chart(emotion_probs): Updates the bar chart with the latest emotion probabilities.
line_chart(emotion_history, duration): Plots a line chart of emotions over time based on the provided emotion history.
main(): Main function to run the program. It sets up the device, loads the pre-trained model, and records and analyzes emotions.


## Supported Emotions
Neutral
Calm
Happy
Sad
Angry
Fearful
Disgust
Surprised

## Customization
You can customize the recording duration and threshold in the main() function by changing the duration and threshold variables, respectively. The duration variable determines the length of each recording segment (in seconds), while the threshold variable sets the average absolute amplitude required to consider a segment as containing speech.

If you want to use a different pre-trained Wav2Vec2 model, update the model_path variable in the main() function with the path to your desired model.

## Troubleshooting
If you encounter issues with PyAudio, ensure that you have the necessary system dependencies installed. For example, on macOS, you may need to install PortAudio using brew install portaudio. On Windows, you may need to install the appropriate PyAudio wheel file for your Python version.

If the program is not detecting your speech, try increasing the threshold variable in the main() function to capture audio at lower amplitudes.

## Limitations
The accuracy of emotion recognition may vary depending on the quality of the pre-trained model used and the specific dataset it was trained on. The model's performance may not be as accurate for certain accents or languages if they were not well-represented in the training data.

Real-time performance may be impacted by the processing capabilities of your system, particularly if you are using a CPU for model inference. If you have a GPU available, ensure that you have the necessary GPU dependencies (e.g., CUDA) installed to leverage its capabilities.
Please feel free to ask if you need any further information or clarification.

#### Maintainers
@TomMcOO1 on Twitter

