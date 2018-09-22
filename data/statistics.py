"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def statistic():
    """After we have all of our videos split between mechanicalmeter and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [mechanicalmeter|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    # data_file = []
    video_number = {}
    # folders = ['train', 'test']
    folders=['/home/sk49/workspace/dataset/QIYI_FACE/train']

    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))

        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.mp4'))
            print(vid_class)
            print(len(class_files))
            video_number[vid_class]=len(class_files)

            # for video_path in class_files:
            #     # Get the parts of the file.
            #     video_parts = get_video_parts(video_path)
            #     #print(video_parts)
            #     train_or_test, classname, filename_no_ext, filename = video_parts
            #

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                # if not check_already_extracted(video_parts):
                #     # Now extract it.
                #     src = os.path.join(train_or_test, classname, filename)
                #     dest = os.path.join(train_or_test, classname,
                #         filename_no_ext + '-%04d.jpg')
                #     call(["ffmpeg","-i",src,"-r"," 1",dest])

                # Now get how many frames it is.
                # nb_frames = get_nb_frames_for_video(video_parts)
                #
                # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
                #
                # print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    # with open('data_file.csv', 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerows(data_file)
    print(video_number)
    with open('number.csv', "w") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in video_number.items():
            csvWriter.writerow([k, v])
        csvFile.close()


def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [mechanicalmeter|test], class, filename, nb frames
    """
    statistic()

if __name__ == '__main__':
    main()
