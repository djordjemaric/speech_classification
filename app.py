import sounddevice as sd
import soundfile as sf
import numpy as np
from tkinter import Tk, Button, Label
from model import Preprocessor, CNNModel
from torch import Tensor

# Global variables for recording
recording_frames = []
stream = None
label = None
preprocessor = Preprocessor()
model = CNNModel()

classes = []
with open('classes.txt', 'r') as f:
    classes = f.read().splitlines()
label_to_class = {}
class_to_label = {}
for idx, cl in enumerate(classes):
    label_to_class[cl] = idx
    class_to_label[idx] = cl



def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    global recording_frames
    recording_frames.append(indata.copy())


def classify_audio(filename):
    # Load your audio and run your model's classification logic
    print(f"Loading audio file {filename} for classification")
    data, fs = sf.read(filename)
    data = Tensor([data])
    wav = preprocessor.preprocess(data)
    print(wav.shape)
    prediction = model.predict_single(wav)
    prediction = class_to_label[prediction.item()]
    print(f"Audio classified as: {prediction}")
    return prediction


def on_button_press(event):
    global stream, recording_frames, label
    recording_frames = []
    stream = sd.InputStream(samplerate=44100, channels=1, callback=audio_callback)
    stream.start()
    label.config(text="Recording...")


def on_button_release(event):
    global stream, recording_frames, label
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    if recording_frames:
        audio_data = np.concatenate(recording_frames, axis=0)
        wav_filename = "recorded.wav"
        sf.write(wav_filename, audio_data, 44100)
        label.config(text="Recording stopped. Classifying...")
        prediction = classify_audio(wav_filename)
        label.config(text=f"Classification: {prediction}")
    else:
        label.config(text="No audio recorded.")


def main():
    global label
    root = Tk()
    root.title("Audio Recorder")

    label = Label(root, text="Press and hold the button to record")
    label.pack(pady=20)

    button = Button(root, text="Hold to Record", width=25, height=5)
    button.bind('<ButtonPress>', on_button_press)
    button.bind('<ButtonRelease>', on_button_release)
    button.pack(pady=20)

    root.mainloop()


if __name__ == '__main__':
    main()