import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#custom imports
from CNNmodel import CustomModel
import dataprepare
import inference

Ravdess = "../ravedess/"

"""
For data merging
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)
data_path.head()
"""

if __name__ == "__main__":
    
    if len(sys.argv) <2:
        print("please provide operation parameter i.e. train or test")
        exit(0)

    opr=sys.argv[1]

    
    if opr == "train":
        #read and prepare dataset
        data=dataprepare.read_ravedess(Ravdess)
        
        #extract features
        FE=dataprepare.FeatureExtract(data_path=data)
        X,Y=FE.load_features()
        Features = pd.DataFrame(X)
        Features['labels'] = Y
        Features.to_csv('resources/features.csv', index=False)
        X = Features.iloc[: ,:-1].values
        Y = Features['labels'].values

        # As this is a multiclass classification problem onehotencoding of the emotions.
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        
        #save encoding for reuse
        with open('resources/model/encoder.pickle', 'wb') as f:
            pickle.dump(encoder, f)

        # splitting data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
        
        #build and train model
        m=CustomModel(x_train.shape[1])
        m.build()
        history=m.train(x_train, y_train,x_test, y_test,batchsize=64,num_epochs=100)
        m.save(filename='resources/model/model.h5')

        #test split
        pred_test = m.predict(x_test)
        y_pred = encoder.inverse_transform(pred_test)
        y_test = encoder.inverse_transform(y_test)
        df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
        df['Predicted Labels'] = y_pred.flatten()
        df['Actual Labels'] = y_test.flatten()

        #plot confusion metrics
        m.plot_confusion_matrix(encoder,x_test,y_test,y_pred)
        #plot loss and accuracy
        m.plot_accuracy_loss(x_test,y_test)


    if opr == "test":
        if len(sys.argv) <3:
            print("please provide audio file path to test")    
            exit(0)
        test_path=sys.argv[2]
        model_file='resources/model/model.h5'    
        if len(sys.argv) >3:
            model_file=sys.argv[3]   
        emotion, emotion_percentage=inference.ser(test_path, model_file)
        
        
        print("\n======================================")        
        print("emotion detected: ", emotion)
        print("emotion accuracy:", emotion_percentage*100 )
        print("======================================\n")
