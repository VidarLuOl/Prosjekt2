import numpy as np
from Activation import Activation

class layer:
    def __init__(self, inputnode, outputnode):
        self.weight = np.random.randn(inputnode, outputnode)
        self.delta = np.random.randn(1, outputnode) 
        self.bias = np.random.randn(1, outputnode)

    def __call__(self, X, act):
        self.z = X@self.weight + self.bias
        if(act == "sigmoid"):
            self.a = Activation.sigmoid(self.z)
            self.da = Activation.derSigmoid(self.z)
        elif(act == "relu"):
            self.a = Activation.relu(self.z)
            self.da = Activation.derRelu(self.z)    
        elif(act == "identity"):
            self.a = Activation.identity(self.z)
            self.da = Activation.derIdentity(self.z)
        else:
            print("En av leddene har ikke riktig aktiveringsfunksjon")
    


class Neural:
    def __init__(self, hl, nodes):
        self.cost = 0
        self.hl = hl
        self.nodes = nodes


    def setUpLayers(self, input):
        inputshape = input[1]

        if(self.hl == 0):
            layers = [layer(inputshape, 1)]
        else:
            layers = []
            layers.append(layer(inputshape, self.nodes))
            for i in range(1, self.hl):
                layers.append(layer(self.nodes, self.nodes))
            layers.append(layer(self.nodes, 1))

        self.layers = layers


    def ForwardProg(self, XD, act):
        self.layers[0](XD, act)

        for i in range(len(self.layers)-2):
            self.layers[i+1](self.layers[i].a, act)

        self.layers[-1](self.layers[-2].a, "identity")

    
    def BackwardProg(self, XD, z, eta, lmbda):
        self.ForwardProg(XD, "relu")
        summ = 0
        self.layers[-1].delta = Activation.derCst(z, self.layers[-1].a)*self.layers[-1].da
        summ += Activation.accuracy(z, self.layers[-1].a)

        for i in reversed(range(1, self.hl)):
            self.layers[i].delta = (self.layers[i+1].delta @ (self.layers[i+1].weight).T) * self.layers[i].da
            
            self.layers[i].weight = self.layers[i].weight - eta * ( self.layers[i-1].a.T @ self.layers[i].delta ) - eta*lmbda*self.layers[i].weight/len(z)

            self.layers[i].bias = self.layers[i].bias - eta * ( self.layers[i].delta[0,:])
        
        self.layers[0].delta = (self.layers[1].delta @ self.layers[1].weight.T) * self.layers[0].da
        self.layers[0].weight = self.layers[0].weight - eta*(XD.T @ self.layers[0].delta) - eta*lmbda*self.layers[0].weight/len(z)
        self.layers[0].bias = self.layers[0].bias - eta*self.layers[0].delta[0,:]

        return np.sum(self.layers[-1].delta)/len(XD), Activation.accuracy(z, self.layers[-1].a)










def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

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