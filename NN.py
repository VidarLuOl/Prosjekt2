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
        elif(act == "derSigmoid"):
            self.a = Activation.derSigmoid(self.z)
        elif(act == "identity"):
            self.a = Activation.identity(self.z)
        else:
            print("En av leddene har ikke riktig aktiveringsfunksjon")


        


class Neural:
    def __init__(self, hl, nodes):
        self.cost = 0
        self.hl = hl
        self.nodes = nodes


    def setUpLayers(self, input):
        """
        Lager en vektor som innholder alle layersene, hver layer innolder 2 verdier, vekter og nodene de vektene peker på. 
        Innholder altså ikke input nodene og verdiene der.
        Kjører bare denne en gang i starten for å sette tingene opp.
        """

        inputshape = input[1]

        if(self.hl == 0): #Lage bare weights mellom input og output
            layers = [layer(inputshape, 1)]
        else: #Lage nodes i hvert hidden layer, også weights mellom layers
            layers = []
            layers.append(layer(inputshape, self.nodes))
            for i in range(1, self.hl):
                layers.append(layer(self.nodes, self.nodes))
            layers.append(layer(self.nodes, 1))

        self.layers = layers


    def ForwardProg(self, XD):
        #Her lager vi hidden nodes utifra weights og input. Vi gjør dette ved å "Call" layer klassen slik at hver klasse som fra før av innholder weights også får vektor klasser kalt a.
        self.layers[0](XD, "derSigmoid")


        for i in range(len(self.layers)-2):
            self.layers[i+1](self.layers[i].a, "derSigmoid")


        self.layers[-1](self.layers[-2].a, "identity")

    
    def BackwardProg(self, XD, z, eta, lmbda):
        self.ForwardProg(XD)
        # self.cost = Activation.derCst(self.layers[-1].a, z)
        self.cost = Activation.loss(self.layers[-1].a, z)
        print(self.cost)

        # W_new = W_old - (eta * dE/dW)  <-- Her vi fikser


        # dE/dW[-1] = dE/dA[-1] * dA[-1]/dZ[-2] * dZ[-2]/dW[-1]    <-- Gradient descent, denne som velger hvilken vei vi går
        # E = Error
        # dE/dA[-1] = (a[-1]-Y)
        # dA[-1]/dZ[-2] = a[-1](1-a[-1])
        # dZ[-2]/dW[-1] = a[-2]
        # dE/dW[-1] = (a[-1]-Y) * a[-1](1-a[-1]) * a[-2]


        # dE/dB[-1] = dE/dA[-1] * dA[-1]/dZ[-2] * dZ[-2]/dB[-1]
        # E = Error
        # dE/dA[-1] = (a[-1]-Y)
        # dA[-1]/dZ[-2] = a[-1](1-a[-1])
        # dZ[-2]/dW[-1] = 1
        # dE/dW[-1] = (a[-1]-Y) * a[-1](1-a[-1]) * 1


        # for i in self.layers:
        #     print(i.delta)

        #delta_L =              (               f'(z_L)                     ) *         dE/dA
        #delta_L =              (        a_L       *         (1-a_L)        ) *       (a_L - z)
        self.layers[-1].delta = (self.layers[-1].a * (1 - self.layers[-1].a)) * (self.layers[-1].a - z)

        #dE/dW[-1] =  dE/dA[-1]  * dA[-1]/dZ[-2]  *    dZ[-2]/dW[-1]
        #dE/dW[-1] = (a[-1] - Y) * a[-1](1-a[-1]) *        a[-2]
        dC_dW =      (self.layers[-2].a).T @ (          self.layers[-1].delta         )# * (self.layers[-2].a).T

        #dE/dB[-1] =  dE/dA[-1]  * dA[-1]/dZ[-2]  *    dZ[-2]/dB[-1]
        #dE/dB[-1] = (a[-1] - Y) * a[-1](1-a[-1]) *        1
        dC_dB =      (         self.layers[-1].delta          )

        # Finner error rate for hvert ledd utenom output
        for i in reversed(range(0, self.hl)):
            #delta_i             = (             w[i+1].T @ delta[i+1]                  ) * (               f'(z_i)                   )   <--- Bytter (w[i+1].T @ delta[i+1]) om slik at regnstykket går opp
            self.layers[i].delta = (self.layers[i+1].delta @ (self.layers[i+1].weight).T) * (self.layers[i].a * (1 - self.layers[i].a))


        # Oppdaterer Bias i hvert ledd
        for i in reversed(range(0, self.hl+1)):
            self.layers[i].bias = self.layers[i].bias - eta * ( self.layers[i].delta)

        for i in reversed(range(1, self.hl+1)):
            self.layers[i].weight = self.layers[i].weight - eta * ( self.layers[i-1].a.T @ self.layers[i].delta )
        self.layers[0].weight = self.layers[0].weight - eta * ( np.array([XD]).T@self.layers[0].delta)














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


        # for i in reversed(range(0, self.hl)): 
            # self.layers[i].delta = (self.layers[i].delta@self.layers[i].weight.T) * (self.layers[-1].a * (1 - self.layers[-1].a))
            # self.layers[i].delta = (self.layers[i+1].delta@(self.layers[i+1].weight).T)#*(self.layers[i].a * (1 - self.layers[i].a))
            
        #     self.layers[i].weight = self.layers[i].weight - (eta * delta_L * self.layers[i-1].a).T

        #     # print(self.layers[i-1].bias)
        #     self.layers[i-1].bias = self.layers[i-1].bias - (eta*delta_L[0,:]).T
        #     # print(self.layers[i-1].bias)
        #     # print()

        # # print("__________________________________________________")

        # delta_L = (delta_L@self.layers[1].weight.T) * (self.layers[0].a * (1 - self.layers[0].a))

        # self.layers[0].weight = self.layers[0].weight - (eta * delta_L * self.layers[0].a)

        # for i in self.layers:
        #     print(i.delta)
        #     print()

        # print()
        # print("______________________________________")
        # print()
        # self.layers[0].bias = self.layers[0].bias - (eta*delta_L[0,:]).T


        # #W_new                  =          W_old         -  eta * dE/dW
        # self.layers[-1].weight  = self.layers[-1].weight - (eta * dE_dW)





        # #dE/dB[-1] =      dE/dA[-1]      *                dA[-1]/dZ[-2]              * dZ[-2]/dB[-1]
        # dE_dB = ((self.layers[-1].a - z) * (self.layers[-1].a*(1-self.layers[-1].a)) * 1)

        # #B_new                =         B_old        -  eta * dE/dB
        # self.layers[-1].bias  = self.layers[-1].bias - (eta * dE_dB)





        #Error til output
        # self.layers[-1].error = (self.layers[-1].a*(1-self.layers[-1].a)) * (z - self.layers[-1].a)

        # for i in range(len(XD[:,0])):
        #     for j in range(self.hl):
        #         tmp = 0
        #         print(np.sum(self.layers[-2].weight*self.layers[-2].error))
        #     print(i)


        
        # for i in range(len(error_hidden)):
        #     tmp = 0
            
        #     lst = []
        #     for j in range(1, len(hidden[0])+1):
        #         tmp = 0
        #         for k in range(len(w2)):
        #             tmp += w2[k][j] * error_output[i]
        #         lst.append(hidden[i][j-1]*(1-hidden[i][j-1])*sum(tmp))
        #     error_hidden[i] = lst

        # for i in self.layers:
        #     print(i.a)
        #     print()

        # for i in self.layers:
        #     print(i.weight)



