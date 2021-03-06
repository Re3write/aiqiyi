import os
import os.path

def get_train_test_lists(version='01'):

    test_file = os.path.join('val_v2.txt')
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

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    data,label=file_groups

      # Do each of our videos.
    group='test'
    for video in range(len(data)):

            # Get the parts.
        classname = label[video]
        filename = data[video]

            # Check if this class exists.
        if not os.path.exists(os.path.join(group, classname)):
            print("Creating folder for %s/%s" % (group, classname))
            os.makedirs(os.path.join(group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.

        oldpath=os.path.join('../../QIYIDATA/IQIYI_VID_VAL/',filename)


        if not os.path.exists(oldpath):
            print("Can't find %s to move. Skipping." % (oldpath))
            continue

        # Move it.
        dest = os.path.join(group, classname, filename)
        print(dest)
        print("Moving %s to %s" % (oldpath, dest))
        os.rename(oldpath, dest)

    print("Done.")

def main():
    """
    Go through each of our mechanicalmeter/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()
    print(group_lists)

    # Move the files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
