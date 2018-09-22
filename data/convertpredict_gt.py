import csv

birth_data = []
with open('train_v2.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        birth_data.append(row)

print(birth_data)

gt_data=[]

for i in range(1, 4935):
    temp=[]
    temp.append(str(i))
    for x in birth_data:
        if (x[1]) == str(i):
            temp.append(x[0])
    gt_data.append(temp)

with open('gt.csv','w',newline='') as f1:
    writer1 = csv.writer(f1)
    writer1.writerows(gt_data)


