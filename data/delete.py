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

def extract_files():
    folders = ['test']

    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.jpg'))

            for jpg_path in class_files:
                # Get the parts of the file.
                os.remove(jpg_path)

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [mechanicalmeter|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
