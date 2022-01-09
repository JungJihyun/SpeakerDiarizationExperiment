
import librosa
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

from keras import Input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np


os.environ["DUDAT_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

train_audio_path = '/media/brlee/data3/ST-AEDS-20180100_1-OS'
labels = os.listdir(train_audio_path)
all_wave = []
all_label = []



def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x


def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)

    shortcut = x

    for i in range(2):
        if (i == 0):
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(2):
        if (i == 0):
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(2):
        if (i == 0):
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(2):
        if (i == 0):
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

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
    input_tensor = Input(shape=(513, 47, 1), dtype='float32', name='input')

    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)

    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(10, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

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
    all_wave = np.array(all_wave).reshape(-1, 513, 47)
    print("all_wave shape: {}".format(all_wave.shape))

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2)

    K.clear_session()

    #with strategy.scope():
    model = get_compiles_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('sv_10_spectogram.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
                         mode='max')

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[mc], batch_size=64, validation_data=(x_val, y_val))

    model.save('sv_10_spectogram.h5')

    model.outputs = Dense(len(labels), activation='softmax')

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train_loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val_loss')

    acc_ax.plot(history.history['accuracy'], 'b', label='train_acc')
    acc_ax.plot(history.history['val_accuracy'], 'g', label='val_acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

main()