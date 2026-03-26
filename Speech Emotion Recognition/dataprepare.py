import pandas as pd
import numpy as np
import os
import sys
import librosa




import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 



class FeatureExtract:

    def __init__(self, data_path, sr=None):
        self.sample_rate=sr
        self.data_path=data_path
    
    def load_features(self):
        print("preparing features .....")
        X, Y = [], []
        for path, emotion in zip(self.data_path.Path, self.data_path.Emotions):
            feature = self.get_features(path)
            for ele in feature:
                X.append(ele)
                # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
                Y.append(emotion)

        return X,Y            

    def extract_features(self, data):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally
        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=self.sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally
        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally
        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=self.sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally    
        return result

    def get_features(self,path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, self.sample_rate = librosa.load(path, duration=2.5, offset=0.6)    
        # without augmentation
        res1 = self.extract_features(data)
        result = np.array(res1)    
        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data)
        result = np.vstack((result, res2)) # stacking vertically    
        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, self.sample_rate)
        res3 = self.extract_features(data_stretch_pitch)
        result = np.vstack((result, res3)) # stacking vertically    
        return result

    def noise(self,data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self,data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self,data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self,data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)



def read_ravedess(Ravdess):
    print("reading ravedess dataset .....")
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        
        if dir == 'audio_speech_actors_01-24':
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

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    return Ravdess_df


def read_crema(Crema):
    crema_directory_list = os.listdir(Crema)
    file_emotion = []
    file_path = []
    for file in crema_directory_list:
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part=file.split('_')
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
    return Crema_df

def read_tess(Tess):
    tess_directory_list = os.listdir(Tess)
    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '/' + file)
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    return Tess_df


def read_surrey(Savee):
    savee_directory_list = os.listdir(Savee)
    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(Savee + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele=='a':
            file_emotion.append('angry')
        elif ele=='d':
            file_emotion.append('disgust')
        elif ele=='f':
            file_emotion.append('fear')
        elif ele=='h':
            file_emotion.append('happy')
        elif ele=='n':
            file_emotion.append('neutral')
        elif ele=='sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    return Savee_df   