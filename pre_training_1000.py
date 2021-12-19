import os
import librosa
import numpy as np
from pathlib import PurePath
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os
os.environ["DUDAT_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


train_audio_path = '/media/brlee/data3/.deep-speaker-wd/LibriSpeech/train-clean-360'
all_wave = []
all_label = []
duration_of_recordings = []
labels = os.listdir(train_audio_path)

def drawDurationofRecodings():
    for label in labels:
        sublabels = os.listdir(train_audio_path + '/' + label)
        for sublabel in sublabels:
            waves = [f for f in os.listdir(train_audio_path + '/' + label + '/' + sublabel) if f.endswith('.flac')]
            for wav in waves:
                samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + sublabel + '/' +wav)
                duration_of_recordings.append(float(len(samples) / sample_rate))
                print("samples {0}, {1}".format(samples, sample_rate))
    plt.hist(np.array(duration_of_recordings))
    plt.show()



def preProcessing():
    for label in labels:
        sublabels = os.listdir(train_audio_path + '/' + label)
        for sublabel in sublabels:
            waves = [f for f in os.listdir(train_audio_path + '/' + label + '/' + sublabel) if f.endswith('.flac')]
            for wav in waves:
                t1 = 0
                t2 = 3000

                file_path = PurePath(train_audio_path + '/' + label + '/' + sublabel + '/' + wav)
                flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
                flac_tmp_audio_data.export(
                    train_audio_path + '/' + label + '/' + sublabel + '/' + file_path.name.replace(file_path.suffix,
                                                                                                   "") + ".wav",
                    format="wav")

                newAudio = AudioSegment.from_wav(
                    train_audio_path + '/' + label + '/' + sublabel + '/' + file_path.name.replace(file_path.suffix,
                                                                                                   "") + ".wav")
                newAudio = newAudio[t1:t2]
                newAudio.export(
                    train_audio_path + '/' + label + '/' + sublabel + '/' + file_path.name.replace(file_path.suffix,
                                                                                                   "") + ".wav",
                    format="wav")


                samples, sample_rate = librosa.load(
                    train_audio_path + '/' + label + '/' + sublabel + '/' + file_path.name.replace(file_path.suffix,
                                                                                                   "") + ".wav",
                    sr=16000)
                samples = librosa.resample(samples, sample_rate, 8000)

                S = librosa.core.stft(samples, n_fft=1024, hop_length=512, win_length=1024)
                S = S.flatten()
                print("len: %d" %len(S))
                if (len(S) == 24111):
                    all_wave.append(S)
                    all_label.append(label)


def get_compiles_model():
    inputs = Input(shape=(24111, 1))

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
    #drawDurationofRecodings()

    preProcessing()

    global all_wave
    print("len(all_wave): {}".format(len(all_wave)))
    print("len(all_label): {}".format(len(all_label)))

    le = LabelEncoder()
    y = le.fit_transform(all_label)
    classes = list(le.classes_)
    print("len(classes) ".format(len(classes)))

    y = np_utils.to_categorical(y, num_classes=len(labels))

    print("len(y): {}".format(len(y)))


    all_wave = np.array(all_wave).reshape(-1, 24111, 1)
    print("shape: {}".format(all_wave.shape))

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                                random_state=777,
                                                shuffle=True)
    #model = get_compiles_model()

    with strategy.scope():
        model = get_compiles_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('pre_training_1000.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=1, validation_data=(x_val, y_val))

    #save weight
    #model.save_weights("pre_training_250_weights")

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


#Run code
#if __name__ == 'main':
main()