#!usr/bin/python
# -*- coding: utf-8 -*-
import csv
import glob
import os
import os.path




def main():

    count=0
    folders = ['train','test']

    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.jpg'))

            for jpg_path in class_files:
                # Get the parts of the file.
                try:
                    with open(jpg_path, 'rb') as fp:
                        data = fp.read()
                        # if (len(data) < 4):
                        #     os.remove(jpg_path)
                        #     count+=1
                        #     print('发现一张')
                        # elif (len(data) < 4096):
                        #     os.remove(jpg_path)
                        #     count+=1
                        #     print('发现一张')
                        if (jpg_path.find('0face')== -1):
                            os.remove(jpg_path)
                            count+=1
                            print('发现一张')
                        else:
                            print('无误')
                except FileNotFoundError:
                    continue
    print('删除'+str(count)+'错误图片')

if __name__ == '__main__':
    main()
