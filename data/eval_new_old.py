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

model = keras.models.load_model(os.path.join('checkpoints', 'Afterwashpart3.018-3.38.hdf5'))

dict_club = {}
with open('dic.csv')as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        dict_club[row[0]] = row[1]

def Eval_Video(mode=None,mcp=None):

    data = get_video()
    total=0
    error_list=[]
    predict_list=[]
    print(data)

    for i in range(len(data)):
        current_video=data[i].split(os.path.sep)[2].split('.')[0]
        predict_videoname=data[i].split(os.path.sep)[2]
        print(current_video)
        pic_path=glob.glob(os.path.join('test','1493','*jpg'))
        final_list=[]
        if(len(pic_path)!=0):
            for x in pic_path:
                if(x.find(current_video) != -1):
                   final_list.append(x)
            if(len(final_list)==0):
                error_list.append(current_video)
            else:
                total+=1
                print(final_list)
                print(time.time())
                predict = Load_predict(final_list)
                predict_list.append([predict_videoname,predict])
                print(time.time())

    with open('predict.csv','w',newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerows(predict_list)


    with open('error_video.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_list)

    print("总计："+str(total)+'  未识别视频'+str(len(error_list)))

def Load_predict(final_list=None):
    # basemodel=torch.load('')
    # MCP=torch.load('')
    result=[]
    max=0
    for x in final_list:
        img = image.load_img(x, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x=x/255.0
        preds = model.predict(x)
        predict=np.argmax(preds)
        result.append(predict)

    count=Counter(result).most_common(1)
    max , _ = count[0]

    return list(dict_club.keys())[list(dict_club.values()).index(str(max))]



def get_video(version='01'):
    video_list=[]
    for x in glob.glob(os.path.join('test','1493','*mp4')):
        video_list.append(x)

    # return video_list[start_num:end_num+1]
    return video_list



if __name__ == '__main__':
    # argv = sys.argv
    # print(argv)
    # start_num = 0
    # end_num =172834   #(172835)
    # start_num = int(argv[1])
    # end_num = int(argv[2])
    # print("start:", start_num, "end", end_num)
    Eval_Video()