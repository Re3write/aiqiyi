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

model = keras.models.load_model(os.path.join('checkpoints', 'Inres.014-2.94.hdf5'))

def Eval_Video(mode=None,mcp=None):

    data, label = get_train_test_lists()
    total=0
    correct=0
    error_list=[]
    print(len(data))
    print(len(label))

    for i in range(len(data)):
        current_video=data[i].split('.')[0]
        current_gt=label[i]
        pic_path=glob.glob(os.path.join('test',current_gt,'*jpg'))
        final_list=[]
        if(len(pic_path)!=0):
            print(current_video)
            for x in pic_path:
                if(x.find(current_video) != -1):
                   final_list.append(x)
            if(len(final_list)==0):
                error_list.append(current_video)
            else:
                total+=1
                print(final_list)
                if(Load_predict(final_list,current_gt)==True):
                    correct+=1


    with open('error_video.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_list)

    print("总计："+str(total)+'   正确：'+str(correct)+'    准确率：'+str(correct/total)+'  未识别视频'+str(len(error_list)))

def Load_predict(final_list=None,current_gt=None):
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

    dict_club = {}
    with open('dic.csv')as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1]


    if list(dict_club.keys())[list(dict_club.values()).index(str(max))] == current_gt:
        return True
    else:
        return False







    # for x in final_list:
    #     data = data.cuda()
    #     data  = Variable(data)
    #     # compute output
    #     output = basemodel(data)
    #     output = MCP(output, target)
    #     _, predicted = torch.max(output.data, 1)
    #     result.append(predicted)
    # count=Counter(result)
    # for key, values in count.items():
    #     re=values+1
    #     break
    #
    # if  re == current_gt:
    #     return True

def get_train_test_lists(version='01'):

    test_file = os.path.join('val.txt')
    test_list=[]
    test_class_list=[]


    with open(test_file) as fin:
        train = [row.strip() for row in list(fin)]
        for row in train:
            temp=row.split(' ')
            for i in range(1,len(temp)):
                test_class_list.append(temp[0])
                test_list.append(temp[i])


    return test_list,test_class_list




if __name__ == '__main__':
    Eval_Video()