# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch


class CarDateSet(data.Dataset):

    def __init__(self, root, lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]

        with open (lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []

        for line in lines:
            imgs.append(os.path.join(root, line.split()[1]))
            labels.append(int(line.split()[2]))

        self.imgs = imgs
        self.labels = labels

        if transforms is None:

            self.transforms = T.Compose([
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                T.Resize((227, 227)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
            ])
        else:
            self.transforms = transforms


    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    dataset = CarDateSet('./data/train', './data/train.txt')
    img, label = dataset[0]  # 相当于调用dataset.__getitem__(0)
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)
