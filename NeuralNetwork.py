import numpy as np
from Activation import Activation

class layer:
    def __init__(self, inputnode, outputnode):
        self.weight = np.random.randn(inputnode, outputnode)
        self.bias = np.random.randn(1, outputnode)

    def __call__(self, X, act):
        self.z = X@self.weight
        if(act == "sigmoid"):
            self.a = Activation.sigmoid(self.z)
        elif(act == "derSigmoid"):
            self.a = Activation.derSigmoid(self.z)
        elif(act == "identity"):
            self.a = Activation.identity(self.z)
        else:
            print("En av leddene har ikke riktig aktiveringsfunksjon")

        # return self.a


class Neural:
    def __init__(self, hl, nodes):
        self.cost = 0
        self.hl = hl
        self.nodes = nodes




    def setUpLayers(self, inputshape):
        """
        Lager en vektor som innholder alle layersene, hver layer innolder 2 verdier, vekter og nodene de vektene peker på. 
        Innholder altså ikke input nodene og verdiene der.
        Kjører bare denne en gang i starten for å sette tingene opp.
        """

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

        for i in self.layers:
            print(i.weight)


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


# def BackwardProg(self, X, y, eta, lmbda):
#         self.ForwardProg(X)

#         self.cost = Activation.cst(y, self.layers[-1].a)
#         diffCost = 0
#         # print(self.cost)
#         """
#         3 blue 1 brown sin måte å forklare cost funksjon

#         Hvordan påvirker weights cost funksjonen?

#         Cost/W_L = z_L/w_L * a_L/z_L * Cost/a_L

#         Cost/a_L = 2(a_L - y)
#         a_L/z_L = aktiverting'(z_L)
#         z_L/w_L = a_(L-1)

#         Cost/W_L = 2(a_L - y) * aktiverting'(z_L) * a_(L-1)             <-- 1 trenings eksempel
#         Cost/W_L = 1/n sum( 2(a_L - y) * aktiverting'(z_L) * a_(L-1) )

        
#         Hvordan påvirker bias cost funksjonen?

#         Cost/b_L = 1 * aktiverting'(z_L) * a_(L-1)             <-- 1 trenings eksempel
#         Cost/b_L = 1/n sum( 1 * aktiverting'(z_L) * a_(L-1) )

#         Hvordan påvirker 

#         Cost/W_L = 2(a_L - y) * aktiverting'(z_L) * a_(L-1)             <-- 1 trenings eksempel
#         Cost/W_L = 1/n sum( 2(a_L - y) * aktiverting'(z_L) * a_(L-1) )
#         """

#         if(self.hl > 0):
#             # for i in self.layers:
#             #     print(i.a)

#             print()
#             print()

#             dCost_da = Activation.derCst(self.layers[-1].a, y)
#             delta_L = (dCost_da * Activation.derSigmoid(self.layers[-1].z))

#             # print(delta_L)
#             # print(self.layers[0].weight.T)
#             # delta_L = (delta_L @ self.layers[0].weight.T)#*Activation.derSigmoid(self.layers[-1].z) 

#             self.layers[-1].weight = self.layers[-1].weight - eta*(self.layers[-2].a.T @ delta_L)
#             # self.layers[-1].bias = self.layers[-1].bias - eta*(self.layers[-2].a.T @ delta_L) - 2*eta*lmbda*self.layers[-1].weight

#             for i in reversed(range(1, len(self.layers)-1)):
#                 delta_L = (delta_L @ self.layers[i+1].weight.T)*Activation.derSigmoid(self.layers[i].z)
#                 self.layers[i].weight = self.layers[i].weight - eta*(self.layers[i-1].a.T @ delta_L)
#                 # diffCost = self.layers[i].weight - eta*(self.layers[i-1].a.T @ delta_L) - 2*eta*lmbda*self.layers[i].weight
#                 # if( diffCost > self.cost):
#                 #     self.layers[i].weight = self.layers[i].weight - eta*(self.layers[i-1].a.T @ delta_L) - 2*eta*lmbda*self.layers[i].weight
#                 # else:
#                 #     self.layers[i].weight = self.layers[i].weight + eta*(self.layers[i-1].a.T @ delta_L) + 2*eta*lmbda*self.layers[i].weight
#                 # self.layers[i].weight = self.layers[i].weight - eta*(self.layers[i-1].a.T @ delta_L) - 2*eta*lmbda*self.layers[i].weight
#                 # self.layers[i].bias = self.layers[i].bias - eta*delta_L[0,:]



#             # delta_L = (delta_L @ self.layers[1].weight.T)*Activation.derSigmoid(self.layers[-1].z)

#             # self.layers[0].weight = self.layers[0].weight - eta*(X.T @ delta_L) - 2*eta*lmbda*self.layers[0].weight


#             for i in self.layers:
#                 print(i.a)

#             # print("COST!", dCost_da)












#         # if(self.hl > 0):
#         #     delta = []
#         #     djdW = []

#         #     delta.append(-(y - self.layers[-1].a)*Activation.derSigmoid(self.layers[-1].a))
#         #     djdW.append((self.layers[-2].a.T)@(delta[0]))

#         #     print(delta)

#             # for i in range(0, self.hl-1):
#             #     # delta.append(delta[i-1]@self.layers[i+1].weight.T*Activation.derSigmoid(self.layers[i].a))
#             #     # djdW.append((self.layers[i-1].a.T)@(delta[hl-i]))
#             #     print(i)
#             #     delta.append((delta[i]@self.layers[1-self.hl-i].weight.T)*Activation.derSigmoid(self.layers[-2-i].z))
#             #     djdW.append(self.layers[-2-i].a.T@delta[-1])

#             # print(djdW[1])
#             # print()
            

#             # delta.append(delta@self.layers[1].weight.T*Activation.derSigmoid(self.layers[0].a))
#             # djdW.append((self.XD.T)@(delta[]))

#             # delta.append(-(self.z - self.layers[-1].a)*Activation.derSigmoid(self.layers[-1].a))
#             # djdW.append((self.layers[-2].a.T)@(delta[0]))

#             # delta.append(delta[0]@self.layers[2].weight.T*Activation.derSigmoid(self.layers[1].a))
#             # djdW.append((self.layers[0].a.T)@(delta_3))

#             # delta_2 = delta_3@self.layers[1].weight.T*Activation.derSigmoid(self.layers[0].a)
#             # djdW1 = (self.XD.T)@(delta_2)



#         #     delta4 = -(self.z - self.layers[-1].a)*Activation.derSigmoid(self.layers[-1].z)
#         #     djdW3 = (self.layers[-1].a.T)@(delta4)

#         #     delta3 = (delta4@self.layers[-1].weight.T)*Activation.derSigmoid(self.layers[-2].z)
#         #     djdW2 = (self.layers[-2].a.T)@delta3

#         #     delta2 = (delta3@self.layers[-2].weight.T)*Activation.derSigmoid(self.layers[-3].z)
#         #     djdW1 = (self.XD.T)@(delta2)

 


#         # print(djdW3)
#         # print(djdW2)
#         # print(djdW1)


#         # cost_1 = Activation.cst(self.z, self.layers[-1].a)

#         # skalar = 3

#         # for i in self.layers:
#         #     # print(i.weight)
#         #     print(i.a)

#         # self.layers[2].weight = self.layers[2].weight + skalar*djdW[2]
#         # self.layers[1].weight = self.layers[1].weight + skalar*djdW[1]
#         # self.layers[0].weight = self.layers[0].weight + skalar*djdW[0]

#         # self.ForwardProg()
#         # for i in self.layers:
#         #     # print(i.weight)
#         #     print(i.a)


#         # self.layers[2].weight = self.layers[2].weight - skalar*djdW3
#         # self.layers[1].weight = self.layers[1].weight - skalar*djdW2
#         # self.layers[0].weight = self.layers[0].weight - skalar*djdW1

#         # # # self.ForwardProg()
#         # for i in self.layers:
#         #     print(i.weight)
#         #     # print(i.a)



#         # # self.ForwardProg()

#         # cost_2 = Activation.cst(self.z, self.layers[-1].a)

#         # print(cost_1, cost_2)