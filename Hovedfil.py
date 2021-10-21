from Funksjoner import *

#For FrankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


"""_________________________Variabler_______________________"""
n = 20          #Antall punkter i x- og y-retning på modellen
noise = 0.1     #Hvor mye støy påvirker modellen 
p = 10          #Grad av polynom
s = 0.3         #Hvor stor del av dataen som skal være test
conf = 1.96     #Confidence intervall, 95% her
lamda = 1e-10   #Swag master 3000 verdier lizm

"Hvilken oppgave vil du kjøre"
Opg = 1        #Hvilken Opg som skal kjøre, 1 = OLS, 2 = Bootstrap, 3 = CV
ridge = False   #Om du vil inkludere Ridge Regression på Frankefunksjonen
prnt = 1        #Om du vil printe ut resultater. 0=nei, 1=ja
plot = 1        #Om du vil plotte ut resultater. 0=nei, 1=ja

"""_________________________________________________________"""


# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)

x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


def Opg1():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    OLS(x, y, z, p, n, s, conf, lamda, prnt, plot, ridge)


if __name__ == "__main__":
    if(Opg == 1):
        Opg1()
    else:
        print("Du må velge en oppgave!")

