import numpy as np
#import minpy.numpy as np
import pickle as pickle
from sklearn import metrics
from operator import itemgetter
from itertools import groupby
from collections import Counter
import time
from multiprocessing import Pool, cpu_count
import threading


# class Compare(threading.Thread):
#     def __init__(self, feats, video_name):
#         threading.Thread.__init__(self)
#         self.feats = feats
#         self.video_name = video_name
#
#     def run(self):
#         #print("1111111111111111111111")
#         #print(self.feats.shape())
#         #print(features.shape())
#         try:
#             #print(len(self.feats))
#             #print(len(self.feats[0]))
#             #print(len(features))
#             #print(len(features))
#             dis = metrics.pairwise.cosine_similarity(self.feats, features)
#             #print("22222222222222222222")
#             nearest_ids = np.argsort(-dis)[:, :10]
#             #print("3333333333333333333")
#             index = nearest_ids.reshape(-1)
#             #print("4444444444444444444444444")
#             classes_temp = classes[index]
#             #print("555555555555555555555")
#             class_count = Counter(classes_temp)
#             predict_result = int(class_count.most_common(1)[0][0])
#             #write.write(self.video_name + " " + str(predict_result))
#             #print("66666666666666")
#             print(self.video_name, predict_result)
#             #print("7777777777777777")
#         except Exception as e:
#             print("error------", e)
#
#
# def similarity_test(videoNames):
#     feats = test_dict[videoNames]
#     if len(feats) != 0:
#         curThread = Compare(feats=feats, video_name=videoNames)
#         curThread.start()
#         curThread.join(timeout=10)
#     #print(videoNames, "start")
#
#
#


if __name__ == '__main__':
    #videoNames = test_dict.keys()
    #videoNames = videoNames[:1000]
    #print(videoNames)
    print("cpu_count", cpu_count())
    #pool = Pool(cpu_count())
    pool = Pool(10)
    start = time.time()
    pool.map(similarity_test, test_videos)
    pool.close()
    end = time.time()
    print("time", end - start)
    print("complete")

    # result = []
    # result_file = open("./result.txt", 'w')
    # result = similarity_test(test_path, truth_path)
    # result.sort(key=itemgetter(1))
    # groups = groupby(result, itemgetter(1))
    # # seq = [["A", 0], ["B", 1], ["C", 0], ["D", 2], ["E", 2]]
    # # seq.sort(key=itemgetter(1))
    # # groups = groupby(seq, itemgetter(1))
    # for key, data in groups:
    #     list = [item[0] for item in data]
    #     video_names = (' ').join(list)
    #     result_file.write(str(key) + " " + video_names + "\n")
    print("done!")
    #print(time.time() - start)
