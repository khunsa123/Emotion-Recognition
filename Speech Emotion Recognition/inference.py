import pickle
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from dataprepare import FeatureExtract

def preprocess_data(test_file_path):
  x=[]
  FE=FeatureExtract(test_file_path)
  feature = FE.get_features(test_file_path)
  for f in feature:
    x.append(f)
  features = pd.DataFrame(x) 
  x = features.iloc[: ,:].values
  
  # scaling our data with sklearn's Standard scaler
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  return x


def ser(audio_data, model_file):
  x_test=preprocess_data(audio_data)
  model = keras.models.load_model(model_file)
  pred = model.predict(x_test)

  with open('resources/model/encoder.pickle', 'rb') as f:
    enc = pickle.load(f)
  emotion_detected = enc.inverse_transform(pred).flatten()

  # calculate percentage of emotion
  emotion_percentage = [[i,r] for i,r in enumerate(pred[0])]
  emotion_percentage.sort(key=lambda x: x[1], reverse=True)
  
  return emotion_detected[0], emotion_percentage[0][1]
