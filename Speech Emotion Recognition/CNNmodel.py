from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib as plt


class CustomModel:

    def __init__(self, x_train_shape,k_size=5,p_size=5,num_classes=8,batch_size=64, loss = 'categorical_crossentropy',opt = 'adam' , metrics = ['accuracy'] ):
        
        
        self.x_train_shape=x_train_shape
        self.k_size=k_size
        self.p_size=p_size
        self.num_classes=num_classes
        self.batch_size=batch_size
        self.loss=loss
        self.opt=opt
        self.metrics=metrics
        self.model=None
        self.history=None

    def loss_callback(self):
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=1, patience=2, min_lr=0.0000001)
        return rlrp

    def build(self):                
        self.model=Sequential()
        self.model.add(Conv1D(256, kernel_size=self.k_size, strides=1, padding='same', activation='relu', input_shape=(self.x_train_shape, 1)))
        self.model.add(MaxPooling1D(pool_size=self.p_size, strides = 2, padding = 'same'))

        self.model.add(Conv1D(256, kernel_size=self.k_size, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.p_size, strides = 2, padding = 'same'))

        self.model.add(Conv1D(128, kernel_size=self.k_size, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.p_size, strides = 2, padding = 'same'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(64, kernel_size=self.k_size, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.p_size, strides = 2, padding = 'same'))

        self.model.add(Flatten())
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(units=self.num_classes, activation='softmax'))
        self.model.compile(optimizer = self.opt , loss = self.loss , metrics = self.metrics)

    def train(self, x_train,y_train,x_test,y_test, batchsize=64,num_epochs=100):
        print("model training start.....")
        history= self.model.fit(x_train, y_train, batch_size=batchsize, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[self.loss_callback()])
        self.history=history
        return self.history

    def get_model(self):
        return self.model 

    def save(self, filename='model.h5'):
        self.model.save(filename)

    def plot_accuracy_loss(self,x_test,y_test):
        print("Accuracy of our model on test data : " , self.model.evaluate(x_test,y_test)[1]*100 , "%")
        epochs = [i for i in range(50)]
        fig , ax = plt.subplots(1,2)
        train_acc = self.history.history['accuracy']
        train_loss = self.history.history['loss']
        test_acc = self.history.history['val_accuracy']
        test_loss = self.history.history['val_loss']

        fig.set_size_inches(20,6)
        ax[0].plot(epochs , train_loss , label = 'Training Loss')
        ax[0].plot(epochs , test_loss , label = 'Testing Loss')
        ax[0].set_title('Training & Testing Loss')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")

        ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
        ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
        ax[1].set_title('Training & Testing Accuracy')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        fig.savefig('resources/loss_accuracy.png', dpi=fig.dpi)
        plt.show()

    def plot_confusion_matrix(self,encoder,x_test,y_test,y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig=plt.figure(figsize = (12, 10))
        cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
        sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        fig.savefig('resources/confusion_matrix.png', dpi=fig.dpi)
        plt.show()
        print(classification_report(y_test, y_pred))    


