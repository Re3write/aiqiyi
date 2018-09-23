# /home/sk49/workspace/dataset/QIYI_FACE/Test/leak

import csv
import glob
import os
import os.path
from subprocess import call
from multiprocessing import Pool

def extract_files(class_files):

        video_path =class_files
        # Get the parts of the file.
        # print(video_parts)
        filename = video_path[0]
        filename_no_ext = video_path[0].split('.')[0]
        video_parts=filename_no_ext,filename
        print(video_parts)

        # Only extract if we haven't done it yet. Otherwise, just get
        # the info.
        if not check_already_extracted(video_parts):
            # Now extract it.
            src = os.path.join('/home/sk49/workspace/dataset/character_recog_iqiyi/test/IQIYI_VID_TEST',filename)
            dest = os.path.join('/home/sk49/workspace/dataset/QIYI_FACE/Test/leak',
                                filename_no_ext + '-%04d.jpg')
            call(["ffmpeg", "-i", src, "-r", " 1", dest])

        # Now get how many frames it is.
        nb_frames = get_nb_frames_for_video(video_parts)


        print("Generated %d frames for %s" % (nb_frames, filename_no_ext))



def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join('/home/sk49/workspace/dataset/QIYI_FACE/Test/leak',
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[8]
    filename_no_ext = filename.split('.')[0]

    return filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join('/home/sk49/workspace/dataset/QIYI_FACE/Test/leak',
                               filename_no_ext + '-0001.jpg')))



if __name__ == '__main__':
    class_files=[]
    # class_files = glob.glob(os.path.join('/home/sk49/workspace/dataset/character_recog_iqiyi/test/IQIYI_VID_TEST','*.mp4'))
    with open('unread_video.csv','r') as f:
        reader=csv.reader(f, delimiter=',')
        for row in reader:
            class_files.append(row)

    pool=Pool(10)
    pool.map(extract_files,class_files)
    pool.close()
    pool.join()
    print("done")
