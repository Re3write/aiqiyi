import keras
from keras.models import Sequential
from keras.layers import Dropout,Dense,Softmax,BatchNormalization,Activation
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import pickle
import numpy as np
from keras.utils import to_categorical
import os

checkpoint = ModelCheckpoint('data/checkpoints/feats_netnodense_two1024.{epoch:03d}-{val_loss:.2f}.hdf5')

earlyStopping = EarlyStopping(monitor='val_acc',patience=5)

relr = ReduceLROnPlateau(factor=0.5,patience=2)

def get_feats():
    if os.path.isfile("meter_train.npz"): #if we save the data in the form of npz before ,read and rerturn it
        data = np.load("meter_train.npz")
        print('Loading training data from ' + "meter_train.npz")
        return data['image_data'], data['label']


    image_data=[]
    label=[]
    with open('feats_train_sample1000.pickle', 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='bytes')
    for key in  feats_dict:
        for x in feats_dict[key]:
            image_data.append(x)
            label.append(int(key))

    image_data=np.array(image_data)
    label=to_categorical(label)
    # print(image_data.shape)
    np.savez("meter_train.npz", image_data=image_data, label=label)

    return image_data,label

def get_val():
    if os.path.isfile("meter_val.npz"): #if we save the data in the form of npz before ,read and rerturn it
        data = np.load("meter_val.npz")
        print('Loading training data from ' + "meter_val.npz")
        return data['image_data'], data['label']

    image_data=[]
    label=[]
    with open('feats_val_sample1000.pickle', 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='bytes')
    for key in  feats_dict:
        for x in feats_dict[key]:
            image_data.append(x)
            label.append(int(key))

    val_image=np.array(image_data)
    val_label=to_categorical(label)
    print(val_label.shape)

    np.savez("meter_val.npz", image_data=val_image, label=val_label)
    return val_image,val_label

def feats_net():
    image,label =get_feats()
    val_image,val_label=get_val()

    model= Sequential()
    # model.add(Dense(5012,input_shape=(512,)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Dense(1024,input_shape=(512,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4935))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy','top_k_categorical_accuracy'])

    model.fit(image,label,batch_size=64,epochs=30,shuffle=True,validation_data=(val_image,val_label),callbacks=[checkpoint,earlyStopping,relr])

if __name__ == '__main__':
    feats_net()





