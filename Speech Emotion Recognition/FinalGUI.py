import os
import base64
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pathlib
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# to play the audio files
from IPython.display import Audio
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
#from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pickle
from PIL import ImageTk, Image
import wave
import pygame



#system and resource information
print("Running tensorflow version: {}".format(tf.keras.__version__))
print("Running tensorflow.keras version: {}".format(tf.__version__))
print("Running keras version: {}".format(keras.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

import warnings

#systems warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


Ravdess = "Ravdess/"
Crema = "crema/AudioWAV/"
Tess = "tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "surrey/ALL/"

print("Reading Ravdess")
ravdess_directory_list = os.listdir(Ravdess)
file_emotion = []
file_path = []
for dir in ravdess_directory_list:

    if dir == 'audio_speech_actors_01-24':
        print("skip diirectory -- : " + dir)
        continue
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)


# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

Ravdess_df.Emotions.replace(
    {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

#removing emotions

Ravdess_df = Ravdess_df[Ravdess_df != 'fear']
Ravdess_df = Ravdess_df[Ravdess_df != 'disgust']
Ravdess_df = Ravdess_df[Ravdess_df != 'calm']
Ravdess_df = Ravdess_df[Ravdess_df['Emotions'].notna()]


print("Reading Crema")

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')


# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()

Crema_df = Crema_df[Crema_df != 'fear']
Crema_df = Crema_df[Crema_df != 'disgust']
Crema_df = Crema_df[Crema_df != 'Unknown']
Crema_df = Crema_df[Crema_df['Emotions'].notna()]


print("Reading Tess")

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()
Tess_df = Tess_df[Tess_df != 'fear']
Tess_df = Tess_df[Tess_df != 'disgust']
Tess_df = Tess_df[Tess_df['Emotions'].notna()]

print("Reading Savee")

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()
Savee_df = Savee_df[Savee_df != 'fear']
Savee_df = Savee_df[Savee_df != 'disgust']
Savee_df = Savee_df[Savee_df['Emotions'].notna()]





# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)
data_path.to_csv("data_path.csv", index=False)
data_path.head()



emotion = 'angry'
path = np.array(Ravdess_df.Path[Ravdess_df.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
# create_waveplot(data, sampling_rate, emotion)
# create_spectrogram(data, sampling_rate, emotion)
Audio(path)


#data augmentation

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)



"""## Data preparation"""

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


#get features

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result



# pre-process data
def preprocess_data(test_file_path):
    x = []
    feature = get_features(test_file_path)
    for f in feature:
        x.append(f)
    features = pd.DataFrame(x)
    x = features.iloc[:, :].values

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = np.expand_dims(x, axis=2)
    return x
#


def ser(audio_data, model_file):
  x_test=preprocess_data(audio_data)
  model = keras.models.load_model(model_file)
  pred = model.predict(x_test)

  with open('encoderAllData.pickle','rb') as f:
      enc = pickle.load(f)
  emotion_detected = enc.inverse_transform(pred).flatten()

  emotion_percentage = [[i, r] for i, r in enumerate(pred[0])]
  emotion_percentage.sort(key=lambda x: x[1], reverse=True)
  global emotion
  emotion = emotion_detected[0]
  global accuracy
  accuracy = str(emotion_percentage[0][1])
  # print(type(emotion))
  # print(type(accuracy))

  return emotion, accuracy



####BACK END CODE ###############
angry_pic = 'angry.png'
sad_pic = 'sad.png'
happy_pic = 'happy.png'
neutral_pic = 'neutral.png'
surprise_pic = 'surprise.png'

spectogram_pic = None



model = []
audio = []
spectogram = []

def load_model():
    try:
        model_path = filedialog.askopenfilename()
        model.append(model_path)
    except FileNotFoundError:
        print("File Not Found")


def load_audio():
    try:

        audio_path = filedialog.askopenfilename()
        audio.append(audio_path)
        audio_file = audio[-1]
        wav = wave.open(audio_file, 'r')
        raw = wav.readframes(-1)
        raw = np.frombuffer(raw, "Int16")
        fig = plt.figure()
        plt.figure(figsize=(20, 10))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("waveplot", fontsize=70, fontweight="bold")
        plt.plot(raw, color="blue")
        plt.ylabel("amplitude", fontsize=35, fontweight="bold")
        plt.xlabel("Time ms", fontsize=35, fontweight="bold")
        strFile = "spectogram.png"
        if os.path.isfile(strFile):
            os.remove(strFile)
        plt.savefig(strFile, dpi=32)
        plt.close()
    except FileNotFoundError:
        print("File not found")




def playSound():
    pygame.mixer.init()
    if audio == []:
        messagebox.showinfo("No Audio", "Please Select an Audio File")
    else:
        my_sound = pygame.mixer.Sound(audio[-1])
        my_sound.play()
        pygame.time.wait(int(my_sound.get_length() * 1000))


def show_spectogram():
    if audio == []:
        messagebox.showinfo("No Audio", "Please Select an Audio File")
    else:
        global img
        spectogram_pic = 'spectogram.png'
        img = ImageTk.PhotoImage(Image.open(spectogram_pic))
        canvas1.create_image(316, 156,anchor="center", image=img)



def classify():
    emotion_textbox.delete(1.0,END)
    accuracy_textbox.delete(1.0, END)
    if model == []:
        messagebox.showinfo("No Model","Please Select a Model File")
    if audio == []:
        messagebox.showinfo("No Audio","Please Select an Audio File")

    else:
        prediction = ser(audio[0], model[0])
        emotion_textbox.insert(0.0, emotion)
        accuracy_textbox.insert(0.0,accuracy)
        # accuracy_textbox.insert(0.0, accuracy)
        if emotion ==  "angry":
            display_emotion(angry_pic)
        elif emotion == "sad":
            display_emotion(sad_pic)
        elif emotion == "neutral":
            display_emotion(neutral_pic)
        elif emotion == "happy":
            display_emotion(happy_pic)
        elif emotion == "surprise":
            display_emotion(surprise_pic)

def reset():
    canvas1.delete('all')
    canvas1.configure(bg="darkgrey")
    canvas2.delete('all')
    canvas2.configure(bg="darkgrey")
    emotion_textbox.delete(1.0, END)
    accuracy_textbox.delete(1.0, END)
    audio.clear()



global img1
def display_emotion(picture_path):
        # canvas1 = Canvas(root)
        # canvas1.place(relx=0.64, rely=0.6, width=250,height=250)
        # canvas1.configure(bg="green")
        global img1
        path = picture_path
        img1 = ImageTk.PhotoImage(Image.open(path))
        canvas2.configure(bg="SystemButtonFace")
        canvas2.create_image(120, 120, anchor=CENTER, image=img1)




#root dimensions
dimensions = "800x700"

#emotion frame dimensions
spectogram_frame_width= 400
spectogram_frame_height= 400

#output frame
output_frame_width = 400
output_frame_height = 400

#emotion frame
emotion_frame_width = 200
emotion_frame_height = 200

#Labels frame

label_frame_width = 80
label_frame_height = 60

#button
button_frame_width = 22
button_frame_height = 100
button_width = 15
button_height = 3


root = tk.Tk(className="speech emotion recognition")
root.geometry(dimensions)
root.configure()
root.resizable(False,False)

sound_icon = PhotoImage(file="sound_icon.png")


spectogram_title = Frame(root,height=140, width=300)
spectogram_title.place(relx=0.5,rely=0.07)
spectogram_title.configure(bg="green")
spectogram_label = Label(spectogram_title, text="SPECTROGRAM", font=("Times New Roman",16))
spectogram_label.pack()


canvas1 = Canvas(root, width=600, height=315)
canvas1.configure(bg="darkgrey")
canvas1.place(relx=0.2, rely=0.12)


emotion_title=Frame(root,height=140, width=300)
emotion_title.place(relx=0.72,rely=0.6)
emotion_title.configure(bg="green")
emotion_label = Label(emotion_title,text="EMOTION",font=("Times New Roman",16))
emotion_label.pack()


canvas2 = Canvas(root)
canvas2.place(relx=0.65, rely=0.65, width=240, height=240)
canvas2.configure(bg="darkgrey")


#program title
title = tk.Label(root,text="SPEECH EMOTION RECOGNIZER", )
title.config(width=200,font=("Times New Roman",25))
title.configure()
title.configure(foreground="black")
title.pack()

button_frame = tk.Frame(root,width=button_frame_width,height=button_frame_height)
button_frame.place(relx=0.0,rely=0.12, relwidth=0.22,relheight=0.5)
# button_frame.configure(bg="blue")

player_frame = tk.Frame(root)
player_frame.place(relx=0.24, rely=0.62, width = 200, height=100 )
player_frame.config(bg="red")


#place buttons in the frame
button_load_model = tk.Button(button_frame,text="LOAD MODEL", width=button_width,height=button_height, command=load_model)
button_load_model.pack(side="top")
#button load audio
button_load_audio = Button(button_frame,text="LOAD AUDIO",width=button_width,height=button_height, command=load_audio)
button_load_audio.pack(side="top")

#button spectogram
button_spectrogram = tk.Button(button_frame, text="SPECTOGRAM", width=button_width,height=button_height, command=show_spectogram)
button_spectrogram.pack(side="top")
#button classify
button_classify = tk.Button(button_frame,text="CLASSIFY", width=button_width, height=button_height, command=classify)
button_classify.pack(side="top")


button_reset = tk.Button(button_frame,text="RESET", width=button_width, height=button_height,command=reset)
button_reset.pack(side="top")

button_sound = tk.Button(player_frame,image=sound_icon,width=160,height=80, command=playSound)
button_sound.pack(padx = 10, pady=10)



outPutFrame1 = tk.Frame(root)
outPutFrame1.place(relx=0.0, rely=0.845, height = 150, width=500)
# outPutFrame1.configure(bg="green")
label_emotion = Label(outPutFrame1, text="Prediction", font=("Times New Roman",18))
emotion_textbox = tk.Text(outPutFrame1, font=("Times New Roman",20))
label_emotion.place(x=0, y=0, height=40, width=180)
emotion_textbox.place(x=190,y=0, height=40, width=200)

label_accuracy = Label(outPutFrame1, text="Accuracy", font=("Times New Roman",18))
accuracy_textbox = tk.Text(outPutFrame1, font=("Times New Roman",20))
label_accuracy.place(x=0, y=50, height=40, width=180)
accuracy_textbox.place(x=190,y=50, height=40, width=200)



if __name__=="__main__":
    root.mainloop()
sys.exit()




