import cv2
import logging
import os

logging.basicConfig(level=logging.INFO) #configurando logging

def run():
    #definiendo rutas
    REAL="videos/real/jose.mp4"
    FAKE="videos/fake/jose.mp4"

    #llamando funcion de generacion
    generate(REAL)
    generate(FAKE)

def generate(path):
    img_index=0 #definiendo una variable para el indice
    index=0
    
    logging.info("cargando video")

    cap=cv2.VideoCapture(path) #definiendo ruta del video

    while(cap.isOpened()): #comprobar entrada de datos
        ret,frame=cap.read() #leer los fotogramas

        if ret==False or index>=200: #verifica si el video finalizo o el indice supera la cantidad
            break

        else:

            if index%2==0: #separando unicamente los indices
                #guardamos el frame definiendo su nombre, con el indice y formato
                pr = os.path.sep.join(["./data/real","real_{}.png".format(img_index)])#ruta y nombre de la imagen
                pf = os.path.sep.join(["./data/fake","fake_{}.png".format(img_index)])#ruta y nombre de la imagen

                #haciendo separacion por ruta
                if "videos/real" in path:
                    p=pr
                else:
                    p=pf
                
                cv2.imwrite(p,frame) #guardando imagen
                img_index+=1 #aumetando indice de las imagenes guardadas

        index+=1#incremento del indice general

    logging.info("imagenes guardadas")

    cap.release() #terminando la capa
    cv2.destroyAllWindows() #eliminando cualquier ventana generada


if __name__=='__main__':
    run()