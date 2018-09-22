# /home/sk49/workspace/dataset/QIYI_FACE/Test/leak

import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():
    data_file = []
    class_files = glob.glob(os.path.join('/home/sk49/workspace/dataset/character_recog_iqiyi/test/IQIYI_VID_TEST','*.mp4'))

    for video_path in class_files:
        # Get the parts of the file.
        video_parts = get_video_parts(video_path)
        # print(video_parts)
        filename_no_ext, filename = video_parts

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


    print("Extracted and wrote %d video files." % (len(data_file)))

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

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [mechanicalmeter|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
