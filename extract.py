import csv

temp=[]

with open('crop_result.csv','r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row[1] == str(0):
            print('yes')
            temp.append([row[0]])

with open('unread_video.csv','w',newline='') as f2:
    writer = csv.writer(f2)
    writer.writerows(temp)