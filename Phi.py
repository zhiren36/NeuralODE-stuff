# Phi.py
# neural network to model the potential function
import torch
import torch.nn as nn
import copy
import math

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        # resnet layers
        for i in range(nTh-2):
            self.layers.append(nn.Linear(m,m, bias=True))
        # in the last layer, we need to map from R^m to R^d
        self.layers.append(nn.Linear(m,d, bias=True))
        self.act = antiderivTanh # activation function
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-d,   outputs
        
        We need an ODE layer followed by a linear layer
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh -1):
            x = x + self.h * self.act(self.layers[i](x))
        
        x = self.layers[self.nTh-1].forward(x)

        return x
    # this function computes the Jacobian of the Resnet
    # with respect to the input s = (x,t) and returns the Jacobian with trace
    # note we only need the trace with respect to x
    
    def trace_jac(self, x):
        
        m   = self.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        
        opening     = self.layers[0].forward(x) # K_0 * S + b_0, the opening layer
        # Forward of ResNet N and fill u
        u.append(self.act(opening)) # u0 = \sigma(K_0 * S + b_0)
         # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )
        feat = u[0]
        # the forward process again, where we keep track of all the u's
        for i in range(1,self.nTh-1):
            feat = feat + self.h * self.act(self.layers[i](feat))
            u.append(feat)
        u.append(self.layers[self.nTh-1].forward(feat))
        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        # then we use a for loop to update the Jacobian
        Kopen = self.layers[0].weight[:,0:d+1]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        
        # up to the last layer, jac has dim m by d+1 by nex, but the last layer,
        # jac will have dimension d by d+1 by nex.  
        for i in range(1, self.nTh-1):
            KJ  = torch.mm(self.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            
            temp = self.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            Jac = Jac + self.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian       
        KJ  = torch.mm(self.layers[self.nth-1].weight , Jac.reshape(d,-1))
        KJ  = KJ.reshape(d,-1,nex)
        Jac = KJ + self.layers[self.nth-1].bias[:, None, None] # update Jacobian 
        
        trace = []
        #for i in range(0, len(Jac.shape[2])):
         #   x_jac = Jac[:-1,:,i]
          #  trace.append(torch.trace(x_jac))
        #trace = torch.tensor(trace)
        return jac, trace




if __name__ == "__main__":

    import time
    import math

    # test case
    d = 2
    m = 5

    net = ResNN(nTh=3, m=m, d=d)
    net.layers[0].weight.data  = 0.1 + 0.0 * net.layers[0].weight.data
    net.layers[0].bias.data    = 0.2 + 0.0 * net.layers[0].bias.data
    net.layers[1].weight.data  = 0.3 + 0.0 * net.layers[1].weight.data
    net.layers[1].weight.data  = 0.3 + 0.0 * net.layers[1].weight.data

    # number of samples-by-(d+1)
    x = torch.Tensor([[1.0 ,4.0 , 0.5],[2.0,5.0,0.6],[3.0,6.0,0.7],[0.0,0.0,0.0]])
    y = net(x)
    print(y)

   











