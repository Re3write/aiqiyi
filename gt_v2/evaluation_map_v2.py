# -*- coding: utf-8 -*-
import os

def calculate_map(val_path, my_val_path):
    id2videos = dict()
    with open(val_path, 'rb') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split(' ')
            id2videos[terms[0]] = terms[1:]
    id_num = len(lines)

    my_id2videos = dict()
    with open(my_val_path, 'rb') as fin:
        lines = fin.readlines()
        assert(len(lines) <= id_num)
        for line in lines:
            terms = line.strip().split(' ')
            my_id2videos[terms[0]] = terms[1:]

    ap_total = 0.
    for cid in id2videos:
        videos = id2videos[cid]
        if cid not in my_id2videos:
            continue
        my_videos = my_id2videos[cid]
        # recall number upper bound
        assert(len(my_videos) <= 100)
        ap = 0.
        ind = 0.
        for ind_video, my_video in enumerate(my_videos):
            if my_video in videos:
                ind += 1
                ap += ind / (ind_video + 1)   
        ap_total += ap / len(videos)

    return ap_total / id_num

if __name__ == '__main__': 
    val_path = './gt_val.txt'
    my_val_path = './my_val.txt'
    print( 'mAP: {}'.format(calculate_map(val_path, my_val_path)))
    
