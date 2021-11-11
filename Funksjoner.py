
import numpy as np
from numpy.typing import _128Bit #For FrankeFunction, MSE, R2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS(x, y, z, p, n, s, conf, lamda, prnt, plot, ridge):
    scaler = StandardScaler()
    XD = Design_X(x, y, p) #Designmatrisen blir laget her

    XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=s) #Splitter inn i Train og Test
    scaler.fit(XD_train) #Skalerer
    XD_train_scaled = scaler.transform(XD_train)
    XD_test_scaled = scaler.transform(XD_test)

    XD_train_scaled[:,0] = 1
    XD_test_scaled[:,0] = 1
        
    beta = BetaFunc(XD_train_scaled, z_train, lamda, ridge) #Vi finner beta verdiene her
    for iter in range(10):
        gradients = (2.0/n)*XD_train_scaled.T @ ((XD_train_scaled @ beta)-z_train)
        beta -= 0.1*gradients










"""___________________________________Mindre Funksjoner________________________________"""
def Design_X(x, y, p):
    if len(x.shape)>1:
        x = np.ravel(x)
        y = np.ravel(y)
    
    r = len(x)                  #Antall rader
    c = int((p+1)*(p+2)/2)      #Antall kolonner
    X = np.ones((r, c))         #Lager en matrise hvor alle verdiene er 1

    for i in range(1, p+1):             #looping over the degrees but we have 21 elements
        q = int(i*(i+1)/2)              #Gjør det slik at vi 
        for k in range(i+1):            #this loop makes sure we dont fill the same values in X over and over
            X[:,q+k] = x**(i-k)*y**k    #using the binomial theorem to get the right exponents
    return X

def BetaFunc(X, z, lamda, ridge):
    if(ridge == True):
        return np.linalg.inv(X.T @ X + lamda*np.identity(len(X[0]))) @ (X.T @ z)
    else:
        return np.linalg.inv(X.T @ X) @ (X.T @ z)

def MSE(y, ytilde):
    #MSE er når vi skal sjekke hvor nøyaktig modellen vi har lagd er iforhold til punktene vi har fått inn
    #Jo nærmere 0 jo bedre er MSE, hvis den er 0 så er den mest sannsynlig overfittet pga at normalt vil støy ødellege.
    s = 0
    n = np.size(y)
    s = np.sum((y - ytilde)**2)
    return s/n

def R2(y, ytilde):
    #R2 er en skala mellom 0 og 1, hvor jo nærmere 1 enn er jo bedre er det
    return 1 - np.sum((y - ytilde) ** 2) / np.sum((y - np.mean(y)) ** 2)

def Variance(X, n):
    var = sum((X - np.mean(X))**2) * np.diag(np.linalg.pinv(X.T @ X))
    return var

def ConfInt(conf, variance, n):
    return conf * np.sqrt(abs(variance))/n

"""______________________________________________________________________________________________"""