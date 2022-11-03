# Deep-Learning:Training Hybrid (GD+LS)
import pandas as pd
import numpy   as np
import dlnet   as net
import opt_alg as opt

# Configuration of Deep Laerning
def config_dl():
    dae = np.genfromtxt("cnf_dae.csv", delimiter=",", dtype=None)
    softmax = np.genfromtxt("cnf_clasifier.csv", delimiter=",")
    ######DAE#######
    # dae[0] #iteraciones
    # dae[1] #minibatch
    # dae[2] #lr
    # dae[3] #capas

    ######SFM#####
    # softmax[0] iteraciones
    # softmax[1] lr
    return (
        [np.int32(dae[0]), np.int32(dae[1]), np.float64(dae[2]), list(map(np.int32, dae[3:]))],
        [np.int32(softmax[0]), np.float64(softmax[1])]
    )


#save Deep Learning's weights and cost function 
def save_w_cost(W, Ws, cost):
    W.append(Ws)
    np.savez("w_dl.npz", *W)
    np.savetxt("costo.csv", cost, delimiter=",", fmt="%10f")
    return

# Softmax's train
def train_softmax(x, y, p_sftm):
    _, output_nodes = y.shape  
    w = net.iniWAdam(x.shape[1], output_nodes)
    iteraciones = int(np.random.uniform(p_sftm[0]-5, p_sftm[0]+10))
    #print(w)
    # print(w.shape, v.shape, "w shape, v shape")
    # print (v)
    v = np.zeros(w.shape)
    s = np.zeros(w.shape)
    # print(s.shape,"soy shape de s", s)
    # np.savetxt(f'aw.csv', w, delimiter=',', fmt='%5g')
    # np.savetxt(f'av.csv', v, delimiter=',', fmt='%5g')
    # np.savetxt(f'as.csv', s, delimiter=',', fmt='%5g')
    cost = []
    e = np.random.uniform(10**-6, 10**-8)

    for i in range(iteraciones):
        grad_w, loss = opt.grad_sftm(x, y, w)
        # if i == 0 or i == 1:
        #     loss = loss-(np.random.uniform(0.01, 0.1))
        cost.append(loss)
        #print(grad_w.shape)
        w, v, s = opt.updW_sftm(w, v, s, grad_w, p_sftm[1], 100, e) #fixed value to avoid double scalars, killing gAdam
        # np.savetxt(f'aw.csv{i}', w, delimiter=',', fmt='%5g')
        # np.savetxt(f'av.csv{i}', w, delimiter=',', fmt='%5g')
        # np.savetxt(f'as.csv{i}', w, delimiter=',', fmt='%5g')
        
        
    return (w, cost)

# AE's Train 
def train_dae(x, W, V, p_dae):
    T = int(x.shape[0] / p_dae[1])
    t = 1

    for batch in net.dat_miniBatch(x, p_dae[1]):
        A = net.forward_dae(batch, W)
        gradients = opt.grad_dae(batch, W, A)
        W, V = opt.updW_dae(W, V, gradients, p_dae[2], t, T)
        t += 1

    return W, V

#Deep Learning's Train 
def train_dl(x, p_dae):
    W, V = net.iniW(x.shape[1], p_dae[3])
    

    for _ in range(p_dae[0]):
        W, V = train_dae(x, W, V, p_dae)
    
    W_decoder = []

    for i in range(len(p_dae[3]), len(W))[::-1]:
        W_decoder.append(W[i].T)

    return W_decoder

   
def load_data_trn():
# dtrain,etrain    
    xe = pd.read_csv('dtrain.csv', header = None)
    xe = np.array(xe,dtype=float)
    ye = pd.read_csv('etrain.csv', header = None)
    ye = np.array(ye,dtype=float)
    #xe = np.genfromtxt('dtrain.csv', delimiter=',', skip_header = 0) #dtrain
    #ye = np.genfromtxt('etrain.csv', delimiter=',', skip_header = 0) #etrain
    #xe = np.array(xe)
    
    #ye = np.array(ye)
    # np.random.shuffle(xe)
    # np.random.shuffle(ye)
    #ye = ye.flatten()
    xe = np.transpose(xe)
    ye = np.transpose(ye)
    print(xe.shape,ye.shape, "shape al cargar data")
    return (xe,ye)


# Beginning ...
def main():    
    p_dae, p_sftm = config_dl()
    xe, ye = load_data_trn()
    # print(xe.shape, ye.shape)
    # print(ye.shape)
    W = train_dl(xe, p_dae)
    Xr = net.encoder(xe, W)
    np.savetxt('Xr.csv', Xr, delimiter=',', fmt='%5g')

    Ws, cost = train_softmax(Xr, ye, p_sftm)
    save_w_cost(W, Ws, cost)
       
if __name__ == '__main__':   
	 main()

