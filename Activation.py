import numpy as np

class Activation:
    def cst(z, ztilde):
        return (1.0/len(z))*np.sum((ztilde-z)**2)
        # return np.sum((1/2 * (abs(z - ztilde))**2))/len(z)

    def derCst(z, ztilde):
        return (2.0/len(z))*(ztilde-z)

    def loss(z, ztilde):
        return ((ztilde - z)**2)/2

    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    def derSigmoid(z):
        return np.exp(-z)/((1.0 + np.exp(-z))**2)

    def identity(z):
        return z

    def derIdentity(z):
        return np.ones(z.shape)

    def relu(z):
        return z*(z>0)

    def derRelu(z):
        return 1*(z>0)

    def accuracy(z, ztilde):
        return np.sum((z > 0.5) == ztilde)/len(z)