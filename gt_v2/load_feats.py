# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import time

t0 = time.time()
pickle_path = './feats_train.pickle'
#pickle_path = './feats_val.pickle'

with open(pickle_path, 'rb') as fin:
    feats_dict = pickle.load(fin,encoding='bytes')
print ('features of {} videos'.format(len(feats_dict)))
print ('time: {}'.format(time.time() - t0))

for video_name in feats_dict:
    feats = feats_dict[video_name]
    print ('video {} with {} face features'.format(video_name, len(feats)))
    for feat in feats:
        [frame_num, bbox, det_score, qua_score, feat_arr] = feat
        [x1, y1, x2, y2] = bbox
        print ('frame number: {}'.format(frame_num))
        print ('bounding box: ({}, {}, {}, {})'.format(x1, y1, x2, y2))
        print ('detection confidence: {}'.format(det_score))
        print ('face feature array of dimension {}'.format(len(feat_arr)))
        break
    break

