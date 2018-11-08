import numpy as np



class Neural_Network(object):
    def __init__(self):
        self.listNodes=[2,3,3,1]
        self.nLayers=len(self.listNodes)
        self.Weights=[np.random.randn(y, x) for x, y in zip(self.listNodes[:-1], self.listNodes[1:])]
        self.Bias=[np.random.randn(y, 1) for y in self.listNodes[1:]]
        self.GW = [np.zeros(w.shape) for w in self.Weights]
        self.GB = [np.zeros(b.shape) for b in self.Bias]
        self.eta = 0.01
        self.epochs = 40
        self.pred = 0
        self.E = 0
 
    def Forward_Propagation(self,X):
        Xlist=[X]
        Zlist=[]
        for W, B in zip(self.Weights,self.Bias):
            s=np.transpose([np.dot(W,X)])+B
            X=self.sigmoid(s)
            Xlist.append(np.squeeze(X))
            Zlist.append(np.squeeze(s))
        return Xlist,Zlist

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoidDerivate(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def Back_Propagation(self,X,Y):
        Xlist,Zlist = self.Forward_Propagation(X)
        self.pred=Zlist[-1]
        deltaL=self.Compute_Cost_Deriv(Xlist[-1],Y)*self.sigmoidDerivate(Zlist[-1])
        bderlist= [deltaL]
        wderlist= [np.dot(np.expand_dims(Xlist[-2],1),np.transpose([deltaL]))]
        deltal=deltaL
        for i in range(2,self.nLayers):
            deltal = self.sigmoidDerivate(Zlist[-i])*np.squeeze(np.dot(deltal,np.transpose([self.Weights[-i+1]])))
            bderlist.append(deltal)
            wderlist.append(np.dot(np.transpose([Xlist[-i-1]]),[deltal]))
        return wderlist[::-1],bderlist[::-1]

    def Compute_Cost(self,X,Y):
        for n in range(self.Ldata):
            self.E=self.E+((X-Y[n])**2)/self.Ldata
        return self.E/self.Ldata

    def Compute_Cost_Deriv(self,Yhat,Y):
        return 2*(Yhat-Y)

    def Gradient(self,X,Y):
        for n in range(self.Ldata):
            print(X[n],'sup')
            self.wderlist,self.bderlist=self.Back_Propagation(X[n],Y[n])
            self.GW=[nw+np.transpose(dnw) for nw, dnw in zip(self.GW, self.wderlist)]
            self.GB=[nb+np.transpose([dnb]) for nb, dnb in zip(self.GB, self.bderlist)]
        self.Weights=[w-(self.eta/self.Ldata)*nw for w, nw in zip(self.Weights, self.GW)]
        self.Bias=[b-(self.eta/self.Ldata)*nb for b, nb in zip(self.Bias, self.GB)]

    def Test(self,X,Y):
        error=0
        for n in range(self.Ldata):
            Xlist,Zlist = self.Forward_Propagation(X[n])
            error+=((Xlist[-1]-Y[n])**2)/self.Ldata
        print(error)

    def Train(self,X,Y):
        self.Ldata=len(X)
        for i in range(self.epochs):
            print(i)
            self.Gradient(X,Y)
            self.E=0
            self.Test(X,Y)


sup=np.loadtxt('./datanumerical_train.csv', delimiter=',',skiprows=1)
X=sup[:,1:3]
Y=sup[:,3]
NN=Neural_Network()
NN.Train(X,Y)
#print NN.Gradient(X,Y)
#print yhat

