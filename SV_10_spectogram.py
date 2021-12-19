import os

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["DUDAT_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

train_audio_path = '/media/brlee/data3/ST-AEDS-20180100_1-OS'
labels = os.listdir(train_audio_path)
all_wave = []
all_label = []

def showDurationOfRecordings():
    duration_of_recordings = []

    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
            print("{0}, {1}".format(samples, sample_rate))
            duration_of_recordings.append(float(len(samples) / sample_rate))
        print("duration of recording {}".format(len(duration_of_recordings)))
    plt.hist(np.array(duration_of_recordings))
    plt.show()

def preProcessing():
    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label ) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
            t1 = 0
            t2 = 3000
            #samples = samples[t1:t2]
            newAudio = AudioSegment.from_wav(train_audio_path + '/' + label + '/' + wav)
            newAudio = newAudio[t1:t2]
            newAudio.export(train_audio_path + '/' + label + '/' + wav, format="wav")


            samples = librosa.resample(samples, sample_rate, 8000)

            S = librosa.core.stft(samples, n_fft=1024, hop_length=512, win_length=1024)
            S = S.flatten()
            #print(len(S))
            if (len(S) == 24111):
                all_wave.append(S)
                all_label.append(label)

    print("all wave len: {}".format(len(all_wave)))
    print("all label len: {}".format(len(all_label)))
def get_compiles_model() :
    inputs = Input(shape=(24111,1))

    conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Flatten()(conv)
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)


    model = Model(inputs, outputs)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model

def main():
    #showDurationOfRecordings()
    preProcessing()

    le = LabelEncoder()
    y = le.fit_transform(all_label)
    classes = list(le.classes_)
    print("classes len: {}".format(len(classes)))

    y = np_utils.to_categorical(y, num_classes=len(labels))
    print("y len{}".format(len(y)))

    global all_wave
    all_wave = np.array(all_wave).reshape(-1, 24111, 1)
    print("all_wave shape: {}".format(all_wave.shape))

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                               random_state=777,
                                               shuffle=True)

    K.clear_session()

    #with strategy.scope():
    model = get_compiles_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('sv_10_spectogram.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
                         mode='max')

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

    model.save('sv_10_spectogram.h5')

    model.outputs = Dense(len(labels), activation='softmax')

    #last_layer = model.layers[-1]

   # model = keras.Sequential(last_layer)


   # model.output = Dense(len(labels), activation='softmax')
   #model.

main()