import numpy as np

class Activation:
    def cst(z, ztilde):
        return (1/len(z))*np.sum((z-ztilde)**2)

    def derCst(z, ztilde):
        return (2/len(z))*(z-ztilde)

    def loss(z, ztilde):
        return ((z- ztilde)**2)/2

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def derSigmoid(z):
        return np.exp(-z)/((1 + np.exp(-z))**2)

    def identity(z):
        return z

    def derIdentity(z):
        return np.ones(z.shape)