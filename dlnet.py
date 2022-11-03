# deep Learning Net

import numpy  as np




# Initialize weights of the Deep-AE
def iniW(x_size, hidden_nodes):
    hidden_nodes = [x_size, *hidden_nodes]    
    nodes = hidden_nodes
    
    for node in hidden_nodes[::-1]:
        nodes.append(node)
        
    W, V = [], []
    
    for idx in range(0, len(nodes) - 1):
        w, v = randW(nodes[idx], nodes[idx + 1])
        W.append(w)
        V.append(v)

    return (W, V)

# gets miniBatch
def dat_miniBatch(x, batch_size=1):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, x.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, x.shape[0])
        excerpt = indices[start_idx:end_idx]
        yield x[excerpt]


# Initialize random weights
def randW(next: int, prev: int):
    r = np.sqrt(6.0 / (next + prev))
    w = np.random.rand(next, prev) * 2 * r - r
    return (w, np.zeros(w.shape))

def iniWAdam(next,prev):
    r = np.sqrt(6.0 / (next + prev))
    w = np.random.rand(next, prev) * 2 * r - r
    return(w)

# Feed-forward of AE
def forward_dae(x, W):
    A = [
        x,
    ]
    
    for idx in range(1, len(W) + 1):
        if idx == len(W):
            A.append(sigmoid(A[idx - 1] @ W[idx - 1]))
        else:
            A.append(act_func(A[idx - 1] @ W[idx - 1]))

    return A 

# Encoder
def encoder(x, weigths):
    for w in weigths:
        z = np.dot(x,w)
        x = act_func(z)
    return x

# Activation function
def act_func(z):
    return (2.0 / (1.0 + np.exp(-z))) - 1.0


# Derivate of the activation function
def deriva_func(a):
    return 0.5 * (1.0 - (a ** 2))

# Forward Softmax
def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def deriva_sigmoid(a):
    return a * (1.0 - a)

# Measures
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    np.savetxt('cm.csv', cm, delimiter=",", fmt='%5g')
    Fsc    = fscores(cm)    
    return(cm,Fsc)
 
#Confusion matrix
def confusion_matrix(x,y):
    cm = np.zeros((y.shape[0], x.shape[0]))
    
    for real, predicted in zip(y.T, x.T):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
    return(cm)

# Fscores's function
def fscores(conf_m):
    precision = []

    for i in range(0,conf_m.shape[0]):
        precision.append(conf_m[i,i]/np.sum(conf_m[i]))

    recall = []
    for i in range(0,conf_m.shape[0]):
        recall.append(conf_m[i,i]/np.sum(conf_m[:,i]))

    fscore = []
    for i in range(0,conf_m.shape[0]):
        fscore.append(2*(precision[i])*recall[i]/(precision[i]+recall[i]))

    fscore = np.append(fscore, np.mean(fscore))
    np.savetxt('fscore.csv', fscore, delimiter=",", fmt='%f')
    return(fscore)


# -------------------------------------------------------------------------
#
#
