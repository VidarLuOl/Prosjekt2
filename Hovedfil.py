from NN import *

#For FrankeFunction
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
# from random import seed
import random


"""_________________________Variabler_______________________"""
n = 10          # Hvor mange punkter i terrenget. x^2
noise = 0.1     # Noice når vi lager terreng
nodes = 4       # Antall noder i hver hidden layer
hl = 2          # Antall hidden layer
eta = 0.5       # Learning rate
lmbda = 1       # ??
mb = 5          # Mini-Batch size

"""_________________________________________________________"""

np.random.seed(1337)

# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)

x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


def Opg1():
    
    XD = Design_X(x, y, 1)[:,1:] #1 fordi me skal bare ha x og y verdiene. Aka [1, x, y], eks. ved 3 så funker det men da lager vi undvendig stor matrise [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]. Dette er da kolonne veridene
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=0.7)

    NeuralNetwork = Neural(hl, nodes)

    NeuralNetwork.setUpLayers(np.shape(XD_train))


    for i in range(1, 6):
        NeuralNetwork.BackwardProg(XD_train[i], z_train[i], eta, lmbda)
        print("______________________________")
    




if __name__ == "__main__":
    Opg1()

