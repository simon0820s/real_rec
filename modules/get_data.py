from imutils import paths
import numpy as np
from keras.utils import to_categorical
import logging
import matplotlib.pyplot as plt
import cv2

logging.basicConfig(level=logging.INFO) #estableciendo nivel de los mensajes

def run(train):
    #cargando datos
    logging.info("cargando datos para el entrenamiento")

    REAL_PATH="./data/real"#defieniendo path de las imagenes reales
    FAKE1_PATH="./data/fake_1"#definiendo path de las imganes falsas

    logging.info("cargando y preparando imagenes reales")

    x_r,y_r=get(REAL_PATH)

    logging.info("cargando y preparando imagenes falsas")
    
    x_f1,y_f1=get(FAKE1_PATH)

    x=x_f1+x_r
    y=y_f1+y_r

    logging.info("seleccionando datos correspondientes")

    if train: #separando los datos para entrenamiento
        #crear listas de datos para entrenamiento
        x_train=[]
        y_train=[]
        #llenando las listas con los datos correspondientes

        for i in range(0,len(x)):

            if i%10!=0:# todos los datos son de train excepto 1 de cada 10
                x_train.append(x[i]) #agregando imagen a la lista
                y_train.append(y[i]) #agregando etiqueta a la lista

        x_train=np.array(x_train).astype('float32')/255 #convirtiendo a tensor y normalizando datos a un rango entre 0-1
        y_train=np.array(y_train).astype(int) #convirtiendo a tensor

        logging.info("Datos guardados y separados")

        return x_train,y_train
    
    else:
        #crear listas de datos para entrenamiento
        x_test=[]
        y_test=[]
        #llenando las listas con los datos correspondientes
        for i in range(0,len(x)):
            
            if i%10==0:# todos los datos son de train excepto 1 de cada 10
                x_test.append(x[i]) #agregando imagen a la lista
                y_test.append(y[i]) #agregando etiqueta a la lista

        x_test=np.array(x_test).astype('float32')/255 #conviritiendo a tensor y valores entren 0-1
        y_test=np.array(y_test).astype(int) #convirtiendo a tensor
        logging.info("Datos guardados y separados")

        return x_test,y_test    

def get(path):
    i=0
    imgs=[]
    labels=[]

    if "real" in path:
        label=1
    else:
        label=0

    img_paths=list(paths.list_images(path)) #guardando las rutas de las imagenes

    for i in range(0,len(img_paths)): #bucle para tomar todas las imagenes

        img=cv2.imread(img_paths[i]) #seleccionar la imagen
        img=cv2.resize(img,(100,150))
        img=img[:,:,0]
        imgs.append(img)
        labels.append(label)

    return imgs,labels


if __name__=='__main__':
    run(train=True)