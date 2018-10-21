import numpy as np
import sys
import os
from scipy import *
import cv2
import random
import matplotlib.pyplot as plt

#定义添加高斯噪声的函数
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX, randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if  all(NoiseImg[randX, randY])< 0:
                 NoiseImg[randX, randY]=0
        elif all(NoiseImg[randX, randY])>255:
                 NoiseImg[randX, randY]=255
    return NoiseImg




def load_png_data():
    """Loads a multilayer png file and return a list of all feature loaded as numpy arrays.
    You can use np.concatenate(x,2) to create a 3D array of size"""
    m=1  #训练文件个数
    n=1   #测试文件个数
    train_set_x=[]#训练数据集
    train_set_y=[]#训练标签集

    test_set_x=[]#测试数据集
    test_set_y=[]#测试标签集

    train_data={}

    train_path=r".\dataset\train_label\\"
    dirs=os.listdir(train_path)

    for file in dirs:
        srcImg=cv2.imread(train_path+file)
        #将label数据集保存为numpy格式并保存
        npImg=np.array(srcImg)
        np.save(train_path+str(m)+'.npy',npImg)
        train_set_x.append(npImg)


        NoiseImg = GaussianNoise(srcImg, 25, 4, 0.8)
        npNoiseImg = np.array(NoiseImg)
        cv2.imwrite(r".\dataset\trainset\\"+str(m)+'.png', NoiseImg, [int(cv2.IMWRITE_PNG_STRATEGY_DEFAULT)])
        np.save(r".\dataset\trainset\\" + str(m) + '.npy', npNoiseImg)
        train_set_y.append(npNoiseImg)
        m=m+1
    train_data['train_set_x']=train_set_x
    train_data['train_set_y']=train_set_y

    test_path = r".\dataset\test_label\\"
    dirs_test = os.listdir(test_path)
    for file in dirs_test:
        srcImg=cv2.imread(test_path+file)
        #将label数据集保存为numpy格式并保存
        npImg=np.array(srcImg)
        np.save(test_path+str(n)+'.npy',npImg)
        test_set_x.append(npImg)


        NoiseImg = GaussianNoise(srcImg, 25, 4, 0.8)
        npNoiseImg = np.array(NoiseImg)
        cv2.imwrite(r".\dataset\testset\\"+str(n)+'.png', NoiseImg, [int(cv2.IMWRITE_PNG_STRATEGY_DEFAULT)])
        np.save(r".\dataset\testset\\" + str(n) + '.npy', npNoiseImg)
        test_set_y.append(npNoiseImg)
        n=n+1
    train_data['test_set_x']=test_set_x
    train_data['test_set_y']=test_set_y

    np.savez(r"E:\DeepLearning\CNNDenoiser\dataset\train_data.npz",**train_data)

load_png_data()

