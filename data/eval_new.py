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
import time
from itertools import groupby
from operator import itemgetter
import pickle as pickle

model = keras.models.load_model(os.path.join('checkpoints', 'Afterwashpart3.018-3.38.hdf5'))
image_dir = '/home/sk49/workspace/dataset/QIYI_FACE/Test/face_crop/'
dict_club = {}
with open('dic.csv')as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        dict_club[row[0]] = row[1]


def pic_process(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    return x


def Eval_Video(mode=None,mcp=None):

    data = get_data()
    total=0
    error_list=[]
    predict_list=[]
    # print(data)
    # print(data.keys())
    temp_result = open("./temp_result_ccx.txt", 'w')
    for videoname in data:
        time1 = time.time()
        test_list = [pic_process(image_dir + x) for x in data[videoname]]
        predict = Load_predict(test_list)
        predict_list.append([videoname+".mp4", predict])
        temp_result.write(videoname+".mp4 "+ predict +"\n")
        time2 = time.time()
        print("one video", time2 - time1)

    predict_list.sort(key=itemgetter(1))
    groups = groupby(predict_list, itemgetter(1))
    result_file = open("./result.txt", 'w')
    for key, data in groups:
        list = [item[0] for item in data]
        video_names = (' ').join(list)
        result_file.write(str(key) + " " + video_names + "\n")
    print("done!")



    with open('error_video.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_list)

    print("总计："+str(total)+'  未识别视频'+str(len(error_list)))

def Load_predict(final_list=None):

    img_array=np.array(final_list)
    # print("image",img_array.shape)
    preds = model.predict(img_array,batch_size=img_array.shape[0])
    predict=np.argmax(preds,axis=-1)
    # print("predict",predict)

    count=Counter(predict.flatten()).most_common(1)
    max , _ = count[0]

    return list(dict_club.keys())[list(dict_club.values()).index(str(max))]


def get_data(version='01'):
    out_path = "./video_frame_dic.pickle"
    if os.path.exists(out_path):
        with open(out_path, 'rb') as fin:
            video_dic = pickle.load(fin, encoding="bytes")
            return video_dic

    image_dir = '/home/sk49/workspace/dataset/QIYI_FACE/Test/face_crop/'

    st = time.time()
    video_list = []
    video_dic = {}
    for face_name in os.listdir(image_dir):
        # if count == 20:
        #     break
        video = face_name.split(".")[0]

        video_list.append([face_name,video])
        # count = count + 1
    groups = groupby(video_list, itemgetter(1))
    for key, data in groups:
        image_list = [item[0] for item in data]
        print(image_list)
        video_dic[key] = image_list
    with open(out_path, "wb") as f:
        pickle.dump(video_dic, f)

    end = time.time()
    print("getdata",end -st)

    return video_dic



if __name__ == '__main__':
    # argv = sys.argv
    # print(argv)
    # start_num = 0
    # end_num =172834   #(172835)
    # start_num = int(argv[1])
    # end_num = int(argv[2])
    # print("start:", start_num, "end", end_num)
    Eval_Video()