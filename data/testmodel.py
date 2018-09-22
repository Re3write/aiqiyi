import keras
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import csv

#用opencv读入图片预测会出错


# model=keras.models.load_model(os.path.join('checkpoints','Inres.008-2.91.hdf5'))
# img_path = os.path.join('train','429','0faceIQIYI_VID_TRAIN_00005195-0003.jpg')
# img = image.load_img(img_path, target_size=(196, 196))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# qwer=image.ImageDataGenerator(rescale=1./255)
# x=x/255.0
# preds = model.predict(x)
# result=np.argmax(preds)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # horizontal_flip=True,
    # rotation_range=10.,
    # width_shift_range=0.2,
    # height_shift_range=0.2
)
#
# test_datagen = ImageDataGenerator(rescale=1. / 255,
#                                   )
#
train_generator = train_datagen.flow_from_directory(
    '/home/sk49/workspace/dataset/QIYI_FACE/video/train',
    target_size=(196, 196),
    batch_size=32,
    class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     os.path.join('test'),
#     target_size=(196, 196),
#     batch_size=32,
#     class_mode='categorical')
# print(train_generator.class_indices)
#
#
with open('dic.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(train_generator.class_indices.items())
