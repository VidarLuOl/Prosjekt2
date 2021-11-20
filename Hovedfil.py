from NN import *
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.regularizers import Regularizer, l2
# from keras.optimizers import SGD
import seaborn as sb
import pandas as pd

#For FrankeFunction
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

"""_________________________Variabler_______________________"""
n = 10          # Hvor mange punkter i terrenget. x^2
noise = 0.1     # Noice når vi lager terreng
nodes = 4       # Antall noder i hver hidden layer
batchsize = 1   # Størrelse på batchen
hl = 2          # Antall hidden layer
eta = 0.01      # Learning rate
lmbda = 1       # ??

"""_________________________________________________________"""

# np.random.seed(10)

# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)

x, y = np.meshgrid(x,y)

# breast_data = pd.read_csv("breast-cancer-wisconsin.data", header=None).to_numpy()#, names=["id", "CT", "CSize", "CShape", "Ad", "SECS", "BN", "BC", "NN", "M", "Class"])


noise_full = noise*np.random.randn(n, n)


def FrankeFunc():
    XD = Design_X(x, y, 1)[:,1:] #1 fordi me skal bare ha x og y verdiene. Aka [1, x, y], eks. ved 3 så funker det men da lager vi undvendig stor matrise [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]. Dette er da kolonne veridene
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=0.7)

    index = np.arange(0, len(XD_train))
    Regularizer = [1, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001]
    eta = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.00001]

    mse = np.zeros((len(Regularizer), len(eta)))
    for i, regularizer_ in enumerate(Regularizer):
        for j, eta_ in enumerate(eta):
            NeuralNetwork = Neural(hl, nodes)
            NeuralNetwork.setUpLayers(np.shape(XD_train))
            for k in range(0, 200):
                np.random.shuffle(index)
                XD_train_shuffled = XD_train[index]
                z_train_shuffled = z_train[index]
                for loop in range(0, XD_train_shuffled.shape[0], batchsize):
                    NeuralNetwork.BackwardProg(XD_train_shuffled[loop:loop+batchsize], z_train_shuffled[loop:loop+batchsize], eta_, regularizer_)

            mse[i, j] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta_, regularizer_))

    heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
                                    cbar_kws={'label': 'MSE'},
                                    yticklabels=["%s" %(1*10**-(i)) for i in range(0, len(Regularizer))],
                                    xticklabels=["%s" %(1*10**-(j+1)) for j in range(0, len(eta))],
                                    fmt = ".3",
                                    annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel("eta")
    heatmap.set_ylabel("Regularizer")
    heatmap.invert_yaxis()
    heatmap.set_title("MSE on Franke dataset with FFNN")
    plt.show()

def BreastData():
    # XD = breast_data[:, 1:-1]
    # XD[XD == "?"] = 5
    # XD = XD.astype(int)/10
    # z = breast_data[:, -1]/2-1
    XD = load_breast_cancer().data
    z = load_breast_cancer().target


    XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=0.7)

    index = np.arange(0, len(XD_train))
    Regularizer = [1, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001]
    eta = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.00001]

    mse = np.zeros((len(Regularizer), len(eta)))
    for i, regularizer_ in enumerate(Regularizer):
        for j, eta_ in enumerate(eta):
            NeuralNetwork = Neural(hl, nodes)
            NeuralNetwork.setUpLayers(np.shape(XD_train))
            for k in range(0, 50):
                np.random.shuffle(index)
                XD_train_shuffled = XD_train[index]
                z_train_shuffled = z_train[index]
                for loop in range(0, XD_train_shuffled.shape[0], batchsize):
                    NeuralNetwork.BackwardProg(XD_train_shuffled[loop:loop+batchsize], z_train_shuffled[loop:loop+batchsize], eta_, regularizer_)

            mse[i, j] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta_, regularizer_))

    heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
                                    cbar_kws={'label': 'MSE'},
                                    yticklabels=["%s" %(1*10**-(i)) for i in range(0, len(Regularizer))],
                                    xticklabels=["%s" %(1*10**-(j+1)) for j in range(0, len(eta))],
                                    fmt = ".3",
                                    annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel("eta")
    heatmap.set_ylabel("Regularizer")
    heatmap.invert_yaxis()
    heatmap.set_title("MSE on Breast Cancer with FFNN")
    plt.show()



if __name__ == "__main__":
    FrankeFunc()
    # BreastData()






# def Opg1():
#     XD = Design_X(x, y, 1)[:,1:] #1 fordi me skal bare ha x og y verdiene. Aka [1, x, y], eks. ved 3 så funker det men da lager vi undvendig stor matrise [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]. Dette er da kolonne veridene
#     z = np.ravel(FrankeFunction(x, y) + noise_full)

#     XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=0.7)

#     index = np.arange(0, len(XD_train))
#     # penalties = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
#     # etas = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
#     # epochs = [50, 100, 200, 400, 800, 1000]
#     # p_len = len(penalties)
#     # e_len = len(etas)
#     # ep_len = len(epochs)


#     # MSE_network = np.zeros((p_len, e_len, 2))      # 2 so we can use plot_acc.py
#     # for i, penalty in enumerate(penalties):
#     #     for j, eta in enumerate(etas):
#     #         #Setting up network
#     #         NeuralNetwork = Neural(hl, nodes)
#     #         NeuralNetwork.setUpLayers(np.shape(XD_train))
#     #         # Back-propagation
#     #         NN = []
#     #         for k in range(100):
#     #             np.random.shuffle(index)
#     #             XD_train_shuffled = XD_train[index]
#     #             z_train_shuffled = z_train[index]
#     #             NN.append(abs(NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsize], z_train_shuffled[j:j+batchsize], eta, 0)/batchsize))
#     #         # MSE_network[i,j,0] = mse(y_test, network.feedforward(x_test))
#     #         # MSE_network[i,j,1] = mse(y_train_shuffle, network.feedforward(x_train_shuffle))
#     #         progress = int(100*(p_len*(i+1) + (j+1))/(p_len*p_len + e_len))
#     #         print(f"\r Progress: {progress}%", end = "\r")


#     ting = "eta"


#     if(ting == "batch_epoch"):
#         eta = 0.1
#         mse = np.zeros((3, 10))
#         for batchsizeloop in range(1, 4):
#             for epoch in range(0, 10):
#                 NeuralNetwork = Neural(hl, nodes)
#                 NeuralNetwork.setUpLayers(np.shape(XD_train))
#                 for loop in range(0, 10*epoch):
#                     np.random.shuffle(index)
#                     XD_train_shuffled = XD_train[index]
#                     z_train_shuffled = z_train[index]
#                     for j in range(0, XD_train_shuffled.shape[0], batchsize*batchsizeloop):
#                         NeuralNetwork.BackwardProg(XD_train_shuffled[j:j+batchsizeloop*batchsize], z_train_shuffled[j:j+batchsizeloop*batchsize], eta, 0)

#                 mse[batchsizeloop-1, epoch] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta, 0))

#         heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
#                                         cbar_kws={'label': 'MSE'},
#                                         yticklabels=["%i" %i for i in range(1*batchsize, 4*batchsize, batchsize)],
#                                         xticklabels=["%i" %j for j in range(10, 101, 10)],
#                                         fmt = ".3",
#                                         annot = True)
#         plt.yticks(rotation=0)
#         heatmap.set_xlabel("Epoch")
#         heatmap.set_ylabel("Batchsize")
#         heatmap.invert_yaxis()
#         heatmap.set_title("MSE on Franke dataset with FFNN")
#         plt.show()

#     Regularizer = [1, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001]
#     eta = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.00001]


#     if(ting == "eta"):
#         #Plotter eta mot regularizer(lambda), 
#         mse = np.zeros((len(Regularizer), len(eta)))
#         for i, regularizer_ in enumerate(Regularizer):
#             for j, eta_ in enumerate(eta):
#                 NeuralNetwork = Neural(hl, nodes)
#                 NeuralNetwork.setUpLayers(np.shape(XD_train))
#                 for k in range(0, 200):
#                     np.random.shuffle(index)
#                     XD_train_shuffled = XD_train[index]
#                     z_train_shuffled = z_train[index]
#                     for loop in range(0, XD_train_shuffled.shape[0], batchsize):
#                         NeuralNetwork.BackwardProg(XD_train_shuffled[loop:loop+batchsize], z_train_shuffled[loop:loop+batchsize], eta_, regularizer_)

#                 mse[i, j] = abs(NeuralNetwork.BackwardProg(XD_test, z_test, eta_, 0))

#         heatmap = sb.heatmap(mse[:,:],cmap="viridis_r",
#                                         cbar_kws={'label': 'MSE'},
#                                         yticklabels=["%s" %(1*10**-(i)) for i in range(0, len(Regularizer))],
#                                         xticklabels=["%s" %(1*10**-(j+1)) for j in range(0, len(eta))],
#                                         fmt = ".3",
#                                         annot = True)
#         plt.yticks(rotation=0)
#         heatmap.set_xlabel("eta")
#         heatmap.set_ylabel("Regularizer")
#         heatmap.invert_yaxis()
#         heatmap.set_title("MSE on Franke dataset with FFNN")
#         plt.show()