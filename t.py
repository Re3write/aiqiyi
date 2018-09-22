import keras.preprocessing.image
import os
import os.path
import glob
# import data.move_read_test as dm
# import torch
# from torch.utils import data
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import keras
from collections import Counter
import csv
import sys
from time import time
import cv2


t1= time()
img = image.load_img('frame/IQIYI_VID_TRAIN_00002178.mp4_14.jpg', target_size=(224, 224))
x = image.img_to_array(img)
print(x[:3,:3,:])
t2 = time()
print(t2-t1)
img1 = cv2.imread('frame/IQIYI_VID_TRAIN_00002178.mp4_14.jpg')
img1 = cv2.resize(img1, (224, 224))
print(x[:3,:3,:])
print(time()-t2)


# t=['test\\1\\0faceIQIYI_VID_VAL_0000630-0002.jpg','test\\1\\0faceIQIYI_VID_VAL_0000554-0002.jpg','test\\1\\0faceIQIYI_VID_VAL_0000630-0002.jpg']
#
#
#
# for x in t:
#     if (x.find('IQIYI_VID_VAL_0003023') == -1):
#         print(1)
#         t.remove(x)
# print(t)
#
#
# print('test\\1\\0faceIQIYI_VID_VAL_0000554-0002.jpg'.find('IQIYI_VID_VAL_0003023'))
#
# from collections import Counter
#
# y=[1,1,2,3,4,5,5,5,6,7,7,2,3,3,3,3,3]
#
# o=Counter(y).most_common(1)
# print(o)
# key,_=o[0]
# print(key)
# #
# for key,values in o.items():
#     print(key,values)



import argparse
import os
import time
import torch
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils import data


#
# transform = transforms.Compose([
#     transforms.Resize((112, 96)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
#
# train_dataset = torchvision.datasets.ImageFolder(os.path.join( 'data', 'train'), transform)
# trainloader = data.DataLoader(train_dataset, 16, True)
#
# for batch_idx, (data, target) in enumerate(trainloader):
#     print(target)
#
#
# x=torch.LongTensor([429])
#
# print(x)


# import csv
#
#
# dict_club={}
# with open('data/dic.csv')as f:
#     reader=csv.reader(f,delimiter=',')
#     for row in reader:
#         dict_club[row[0]]=row[1]
#
# print(list(dict_club.keys())[list(dict_club.values()).index('1375')])


# correct=0
# labels=[1,2,3,4,5,6,7]
# predicted=[2,2,3,3,5,5,7]
#
# correct += (predicted == labels).sum().item()
# print(correct)

