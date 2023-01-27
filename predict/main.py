import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from imutils import paths

def run():

    PATH="./predict/imgs" #path de la imagen
    imgs=get_img(PATH) #funcion para recolectar la imagen

    model=tf.keras.models.load_model("./models/trained_model.h5") #carga del modelo entrenado

    result=model.predict(imgs)
    print(result)

    graph(result,imgs) #graficar el resultado

def get_img(PATH):
    logging.info("cargando imagenes")
    imgs=[]
    img_paths=list(paths.list_images(PATH))

    for i in range(0,len(img_paths)):
        img=cv2.imread(img_paths[i])
        img=cv2.resize(img,(100,150))
        imgs.append(img)
    imgs=np.array(imgs).astype(float)/255

    return imgs

def graph(result,imgs):
    logging.info("graficando resultados")
    
    result=list(result)

    rf=[]
    
    for i in result:
        if i>0.98:
            rf.append(1)
        else:
            rf.append(0)
    
    fig=plt.figure(figsize=(5,5))

    plt.subplot(1,2,1)
    plt.imshow(imgs[0],cmap='gray')
    plt.axis('off')
    plt.title(f"categoria: {result[0]}")

    plt.subplot(1,2,2)
    plt.imshow(imgs[1],cmap='gray')
    plt.axis('off')
    plt.title(f"categoria: {result[1]}")

    plt.show()
    
   
if __name__=='__main__':
    run()