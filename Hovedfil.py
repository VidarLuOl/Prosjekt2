from NN import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
# from keras.optimizers import SGD
import seaborn as sb

#For FrankeFunction
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

"""_________________________Variabler_______________________"""
n = 10          # Hvor mange punkter i terrenget. x^2
noise = 0.1     # Noice når vi lager terreng
nodes = 4       # Antall noder i hver hidden layer
batchsize = 5  # Størrelse på batchen
hl = 2          # Antall hidden layer
eta = 0.01       # Learning rate
lmbda = 1       # ??
mb = 5          # Mini-Batch size

"""_________________________________________________________"""

# np.random.seed(10)

# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)

x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


def Opg1():
    
    XD = Design_X(x, y, 1)[:,1:] #1 fordi me skal bare ha x og y verdiene. Aka [1, x, y], eks. ved 3 så funker det men da lager vi undvendig stor matrise [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]. Dette er da kolonne veridene
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=0.7)

    # index = np.arange(0, len(XD_train))
    # np.random.shuffle(index)
    # XD_train_shuffled = XD_train[index]
    # z_train_shuffled = z_train[index]
    # NeuralNetwork = Neural(hl, nodes)
    # NeuralNetwork.setUpLayers(np.shape(XD_train))
    # NN = []
    # eta = 0.1
    # for j in range(0,10,3):
    #     print(XD_train_shuffled[j:j+batchsize])
    #     print(z_train_shuffled[j:j+batchsize])
    #     NN.append(abs(NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsize], z_train_shuffled[j:j+batchsize], eta, 0)/batchsize))

    # print(NN)




    index = np.arange(0, len(XD_train))
    # penalties = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
    # etas = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    # epochs = [50, 100, 200, 400, 800, 1000]
    # p_len = len(penalties)
    # e_len = len(etas)
    # ep_len = len(epochs)


    # MSE_network = np.zeros((p_len, e_len, 2))      # 2 so we can use plot_acc.py
    # for i, penalty in enumerate(penalties):
    #     for j, eta in enumerate(etas):
    #         #Setting up network
    #         NeuralNetwork = Neural(hl, nodes)
    #         NeuralNetwork.setUpLayers(np.shape(XD_train))
    #         # Back-propagation
    #         NN = []
    #         for k in range(100):
    #             np.random.shuffle(index)
    #             XD_train_shuffled = XD_train[index]
    #             z_train_shuffled = z_train[index]
    #             NN.append(abs(NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsize], z_train_shuffled[j:j+batchsize], eta, 0)/batchsize))
    #         # MSE_network[i,j,0] = mse(y_test, network.feedforward(x_test))
    #         # MSE_network[i,j,1] = mse(y_train_shuffle, network.feedforward(x_train_shuffle))
    #         progress = int(100*(p_len*(i+1) + (j+1))/(p_len*p_len + e_len))
    #         print(f"\r Progress: {progress}%", end = "\r")


    ting = "YOLO"
    if(ting == "batch_epoch"):
        eta = 0.1
        mse = np.zeros((3, 10))
        for batchsizeloop in range(1, 4):
            for epoch in range(0, 10):
                NeuralNetwork = Neural(hl, nodes)
                NeuralNetwork.setUpLayers(np.shape(XD_train))
                for loop in range(0, 10*epoch):
                    np.random.shuffle(index)
                    XD_train_shuffled = XD_train[index]
                    z_train_shuffled = z_train[index]
                    for j in range(0, XD_train_shuffled.shape[0], batchsize*batchsizeloop):
                        NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsizeloop*batchsize], z_train_shuffled[j:j+batchsizeloop*batchsize], eta, 0)

                mse[batchsizeloop-1, epoch] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta, 0))

        heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
                                        cbar_kws={'label': 'MSE'},
                                        yticklabels=["%i" %i for i in range(5, 16, 5)],
                                        xticklabels=["%i" %j for j in range(10, 101, 10)],
                                        fmt = ".3",
                                        annot = True)
        plt.yticks(rotation=0)
        heatmap.set_xlabel("Epoch")
        heatmap.set_ylabel("Batchsize")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with FFNN")
        plt.show()

    if(ting == "eta"):
        eta = 0.1
        mse = np.zeros((3, 10))
        for batchsizeloop in range(1, 4):
            for epoch in range(0, 10):
                NeuralNetwork = Neural(hl, nodes)
                NeuralNetwork.setUpLayers(np.shape(XD_train))
                for loop in range(0, 10*epoch):
                    np.random.shuffle(index)
                    XD_train_shuffled = XD_train[index]
                    z_train_shuffled = z_train[index]
                    for j in range(0, XD_train_shuffled.shape[0], batchsize*batchsizeloop):
                        NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsizeloop*batchsize], z_train_shuffled[j:j+batchsizeloop*batchsize], eta, 0)

                mse[batchsizeloop-1, epoch] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta, 0))

        heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
                                        cbar_kws={'label': 'MSE'},
                                        yticklabels=["%i" %i for i in range(5, 16, 5)],
                                        xticklabels=["%i" %j for j in range(10, 101, 10)],
                                        fmt = ".3",
                                        annot = True)
        plt.yticks(rotation=0)
        heatmap.set_xlabel("Epoch")
        heatmap.set_ylabel("Batchsize")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with FFNN")
        plt.show() 




if __name__ == "__main__":
    Opg1()