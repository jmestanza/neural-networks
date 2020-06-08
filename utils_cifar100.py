import numpy as np
import pandas as pd 
import os
import pickle
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py


def is_cached_img_array_from_csv(cache_path,base_dir,csv_name):
    data = None
    if not os.path.exists(cache_path):
        file = open(cache_path, 'wb')
        df = pd.read_csv(base_dir+csv_name)
        data = np.zeros((len(df),32,32,3),dtype='uint8')
        for i,imgname in enumerate(df['Image'].values):
            data[i,] = np.array(imread(base_dir+imgname))
        pickle.dump(data, file)
        file.close()
    else:
        file = open(cache_path, 'rb')
        data = pickle.load(file)
        file.close()
    return data

def get_augmented_y(y_t):
    y_aux = [] #[""] * (len(y_train)*4) #y_train.copy()
    for i in range(len(y_t)):
        for j in range(4):
            y_aux.append(y_t[i])
    
    return np.array(y_aux)

def img_data_augmentation(data,y_data):
    new_images = np.zeros((len(data)*4,32,32,3)).astype(np.uint8)

    for i in range(len(data)):
        for j in range(4):
            if j == 0:
                new_images[4*i+j,] = data[i,]
            else:
                new_images[4*i+j,] = np.rot90(new_images[i+j-1,])

    new_labels = get_augmented_y(y_data)

    return new_images, new_labels


def get_y_from_dataframe(base_dir,csv_name):
    df = pd.read_csv(base_dir+csv_name)
    return df['Label'].values



def plot_accuracy(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

