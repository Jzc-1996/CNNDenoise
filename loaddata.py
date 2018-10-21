import os
import numpy as np

def loaddata():
    train_dataset=np.load(r"E:\DeepLearning\CNNDenoiser\dataset\train_data.npz")
    print(train_dataset.files)
    #
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])
    print(train_set_x_orig.shape)
    test_set_x_orig=np.array(train_dataset["test_set_x"][:])
    test_set_y_orig=np.array(train_dataset["test_set_y"][:])
    print(test_set_x_orig.shape)


loaddata()
