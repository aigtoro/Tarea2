
import numpy      as np
import dlnet  as net
import pandas as pd
# Feed-forward of the DL
def forward_dl(x,weigths):
    
    for w in weigths[:-1]:
        z = np.dot(x,w)
        x = net.act_func(z)
    z = np.dot(x,weigths[-1])
    x = net.softmax(z)
    
    return(x) 


def load_w_dl():
    container = np.load("w_dl.npz")
    W = [container[key] for key in container]
    return W


def load_data_tst():
 #'dtest','etest'
    xe = pd.read_csv('dtest.csv', header = None)
    xe = np.array(xe,dtype=float)
    ye = pd.read_csv('etest.csv', header = None)
    ye = np.array(ye,dtype=int)
    xe = np.transpose(xe)
    ye = np.transpose(ye)
    # np.random.shuffle(xe)
    # np.random.shuffle(ye)
    print(xe.shape,ye.shape, "shape al cargar data")
    return (xe,ye)

# Beginning ...
def main():			
	xv,yv  = load_data_tst()
	print(xv.shape,yv.shape)
	W      = load_w_dl()
	zv     = forward_dl(xv,W)
	yv = yv.T
	zv = zv.T
	print(yv.shape, zv.shape , "yv , zv")      		
	cm, Fsc = net.metricas(yv,zv) 		
	print('Fsc-mean {:.5f}'.format(Fsc.mean()))
	

if __name__ == '__main__':   
	 main()
