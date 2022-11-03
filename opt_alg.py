# Algorith of optimization for Deep Learning 

import numpy  as np
import dlnet as net
#-----------------------------------------------------------------------
# Feed-Backward 
def grad_dae(x, W, A):
    deltas = {}
    deltas[len(W)] = (A[len(W)] - x) * net.deriva_sigmoid(A[len(W)])

    for i in range(0, len(W))[::-1]:
        deltas[i] = np.dot(deltas[i + 1],W[i].T) * net.deriva_func(A[i])

    grad_w = [A[i].T @ deltas[i + 1] for i in range(0, len(W))]

    return grad_w  
# Update DAE's Weight
def updW_dae(W, V, gradients, lr, t, T):
    b = 0.9
    tabu = 1.0 - (t / T)
    beta = b * (tabu / ((1.0 - b) + b * tabu))

    for i in range(len(W)):
        w = W[i]
        v = V[i]
        grad_W = gradients[i]

        v = (beta * v) - (lr * grad_W)
        w = w + v

        W[i] = w
        V[i] = v

    return W, V

#Update Decoder's Weight 
def updW_Decoder():    
    ...
    return()

#-----------------------------------------------------------------------
# Softmax's gradient
def grad_sftm(x, y, w):
    z = np.dot(x,w)
    A = net.softmax(z)
    M = y.shape[0]

    cost = (-1/M)*np.sum(np.sum(y*np.log(A)))
    cost = np.float64(cost)
    grad_w = ((-1/M)*np.dot(np.transpose((y-A)),x))
    grad_w = grad_w.T
    return (grad_w, cost)
    
# Update Softmax's Weight
def updW_sftm(w, v,s, grad_w, lr, iteracion,e):
    b1 = 0.9
    b2 = 0.999
    b1 = np.float64(b1)
    b2 = np.float64(b2)

    v = b1*v + (1-b1)*(grad_w)
    s = b2*s + (1-b2)*np.power(grad_w, 2)

    #gAdam = (np.sqrt(1-np.power(b2,iteracion))/(1-np.power(b1,iteracion)+np.finfo(np.float64).eps))*(v/(np.sqrt(s + e)))
    gAdam = (np.sqrt(1-np.power(b2,iteracion))/(1-np.power(b1,iteracion)))*(v/(np.sqrt(s + e)))
    w =  w - lr*gAdam

    return (w, v, s)
#---------------------------------------------------------------------
    
#
