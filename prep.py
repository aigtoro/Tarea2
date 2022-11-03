# Aim:  Create data for training and testing
#       train: dtrain and etrain
#       test : dtest  and etest
from contextlib import nullcontext
from re import I
from tkinter import EXCEPTION
import numpy as np
import pandas as pd
import os

def load_data_raw(fname):
    data = pd.read_csv(fname,header=None)
    data = np.array(data)
    #print(data.shape)
    return(data)

def data_norm(x, xmin, xmax):
    b = 0.99
    a = 0.01
    if xmin == xmax:
        return ((x-xmin)/(xmax-xmin+1e-100))*(b-a)+a
    return ((x-xmin)/(xmax-xmin))*(b-a)+a

def normalizar(data):
    segmentos = data.shape[0]
    for i in range(segmentos):
        data[i,:] = data_norm(data[i,:], np.min(data[i,:]), np.max(data[i,:]))
    return data

def load_param_prep():
    par = np.genfromtxt("cnf_prep.csv",delimiter=',')    
    params=[]    
    params.append(np.int16(par[0])) # Porcentaje de training
    params.append(np.int16(par[1])) # Longitud del segmento
    params.append(np.int16(par[2])) # Cantidad de clases a cargar
    params.append(np.int16(par[3])) # Data a cargar - puede ser data10, data7, data5 ...etc. La carpeta nueva de data debe ser formato "DataX", donde X es el numero.
    return params

def generar_labels(clase_normalizada, clase_actual, cantidad_clases):
    A = clase_normalizada
    matriz = np.zeros((cantidad_clases,A.shape[1]))
    #print(matriz.shape)
    largo = int(A.shape[1])
    for i in range(largo):
        matriz[clase_actual-1, i] = 1 #funciona la raja
    A = np.concatenate((A,matriz), axis=0)
    #np.savetxt(f'matriz{clase_actual}.csv', matriz, delimiter=',', fmt="%d")
    #print(A.shape)
    return(A)


def segment_class(data,params,clase_actual):
    columnas = data.shape[1] #dentro clase hay 4 columnas
    arreglo_segmento = []
    if (data.shape[0] % params[1]):
        entero = int(data.shape[0] / params[1])
        #print (entero, 'soy entero')
        data = data[0:entero*params[1]]
    for columna in range(columnas):
        clase_columna = data[:,columna] # toda la fila
        segmentacion = np.array(np.split(clase_columna, params[1]))
        arreglo_segmento.append(segmentacion)
    clase_segmentada = np.concatenate((arreglo_segmento),axis=1)


    
    #print(clase_segmentada.shape, "soy clase segmentada")
    clase_normalizada = normalizar(clase_segmentada)
    clase_normalizada = generar_labels(clase_normalizada, clase_actual, params[2])
    #np.savetxt(f'clase{clase_actual}.csv', clase_normalizada, delimiter=',', fmt='%5g')
    return(clase_normalizada)

def save_data_toshuffle(dtrain,dtest):
    if not os.path.exists("misc"):
        os.mkdir("misc")
    np.savetxt("misc/shuffletr.csv",dtrain, delimiter=",", fmt="%5g")
    np.savetxt("misc/shufflets.csv",dtest, delimiter=",", fmt="%5g")
    return ()

def data_shuffle(dtrain,dtest):
    train_to_pd = pd.DataFrame(dtrain)
    test_to_pd = pd.DataFrame(dtest)
    train_to_pd = train_to_pd.sample(frac=1, axis=1).reset_index(drop=True)
    test_to_pd = test_to_pd.sample(frac=1, axis=1).reset_index(drop=True)
    test_to_pd = np.array(test_to_pd)
    train_to_pd = np.array(train_to_pd)
    return (train_to_pd, test_to_pd)



def separar_data(dtrain_total,dtest_total, cantidad_clases): #train,test,etrain,etest
    filas_tr_ts = dtrain_total.shape[0]
    limite = filas_tr_ts - cantidad_clases
    dtrain = dtrain_total[:limite,:]
    dtest = dtest_total[:limite,:]
    label_train = dtrain_total[limite:filas_tr_ts,:]
    label_test = dtest_total[limite:filas_tr_ts,:]
    return (dtrain,dtest, label_train, label_test)

def save_data(train,test,etrain,etest):
    np.savetxt("dtrain.csv",train, delimiter=",", fmt="%5g")
    np.savetxt("dtest.csv",test, delimiter=",", fmt="%5g")
    np.savetxt("etrain.csv",etrain, delimiter=",", fmt="%5g")
    np.savetxt("etest.csv",etest, delimiter=",", fmt="%5g")
    return ()

def main():
    #CARGAR PARAMETROS PREP#
    params = load_param_prep()
    #########################
    #OBTENER DIRECTORIOS#####
    carpeta_data = params[3]
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Data{carpeta_data}")
    count = 0
    arreglo_clase_segmentadas = []
    arreglo_labels = []
    cantidad_clases = params[2]
    print(f"cantidad de clases seleccionadas dentro de Data{carpeta_data} para el prep es: ",cantidad_clases)
    dtrain = []
    dtest = []
    for _ in range(cantidad_clases):
        count += 1
        data = load_data_raw(f'Data{carpeta_data}\class{count}.csv')
        #print(data.shape,f' datos para esta clase{count}')
        #############SE OBTIENEN SEGMENTOS##############
        clase_segmentada = segment_class(data,params,count)
        porcentaje = int((clase_segmentada.shape[1]*params[0])/100) #porcentaje de separacion al etiquetar
        clase_segmentada_train = np.array(clase_segmentada[:,:porcentaje])
        clase_segmentada_test = np.array(clase_segmentada[:,porcentaje:])
        dtrain.append(clase_segmentada_train)
        dtest.append(clase_segmentada_test)
        #print("\n ")
    dtrain_total = np.concatenate((dtrain), axis=1)
    dtest_total = np.concatenate((dtest), axis=1)
    #print(dtrain_total)
    ###########SHUFFLE DATA######################
    dtrain_total,dtest_total = data_shuffle(dtrain_total,dtest_total)



    train,test,etrain,etest = separar_data(dtrain_total,dtest_total,cantidad_clases)
    save_data(train,test,etrain,etest)
    #save_data(dtrain_total,dtest_total)


if __name__ == '__main__':   
	 main()