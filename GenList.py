# coding=utf-8
import os
import cv2
import random
import numpy
import sys

# if __name__ == "__main__":

dict = {'Sedan': 0, #设置每一类的名称以及对应的label，名称需要与文件夹名字一致
        'SUV': 1}
rate = 0.1       #随机抽取10%的样本作为验证集
root = './data/train'

Trainlist = []
Testlist = []
alllist = []
index = 0
# max_num = 80000

for folder in dict:
    img_list = [f for f in os.listdir(os.path.join(root, folder)) if not f.startswith('.')]
    for img in img_list:
        str0 = '%d\t%s\t%d\n' % (index, os.path.join(folder, img), dict[folder])
        index += 1
        alllist.append(str0)

random.seed(100)
random.shuffle(alllist)

num = int(len(alllist) * rate)
Testlist = alllist[0:num]
Trainlist = alllist[num:]

Trainfile = open("./data/train.txt", "w")
for str1 in Trainlist:
    Trainfile.write(str1)
Trainfile.close()

Testfile = open("./data/valid.txt", "w")
for str1 in Testlist:
    Testfile.write(str1)
Testfile.close()


