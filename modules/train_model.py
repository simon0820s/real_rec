import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import get_data
import logging

logging.basicConfig(level=logging.INFO)

def run():
    logging.info("cargando datos y modelo")

    x,y=get_data.run(train=True) #funcion de obtener datos

    model=tf.keras.models.load_model('./models/created_model.h5')#funcion de cargar modelo

    x_train,y_train,x_val,y_val=split_data(x,y) #haciendo un split para tener datos de entrenamiento y validacion

    print(f"categorias entrenamiento: {y_train}")

    train(model,x_train,y_train,x_val,y_val) #funcion de entrenamiento

def split_data(imgs,labels):
    logging.info("separando datos de train")

    print(f"logitud completa: {len(imgs)}")

    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    
    for i in range(0,len(imgs)):

        if i%5==0:
            x_val.append(imgs[i])
            y_val.append(labels[i])
        else:
            x_train.append(imgs[i])
            y_train.append(labels[i])
    
    x_train=np.array(x_train).astype('float32')/255
    y_train=np.array(y_train).astype(int)
    x_val=np.array(x_val).astype('float32')/255
    y_val=np.array(y_val).astype(int)

    print(f"logitud validacion: {len(y_val)}")
    print(f"logitud entrenamiento: {len(y_train)}")

    return x_train,y_train,x_val,y_val

def train(model,x_train,y_train,x_val,y_val):

    logging.info("[INFO] comenzando entrenamiento")

    binnacle=model.fit( #entrenamiento del modelo
        x_train,y_train, #datos de entrenamiento con el generador
        batch_size=64, #tamaño de lotes
        epochs=20, #definiendo epocas
        steps_per_epoch=(x_train.shape[0]//64), #defieniendo pasos por epoca
        validation_data=(x_val,y_val)) #iteraciones

    model.save("./models/trained_model.h5")

    logging.info("[INFO] modelo entrenado y guardado")

    graph(binnacle) #graficar

def graph(binnacle):

    logging.info("graficando datos del entrenamiento")
    
    #definiendo variables
    history_dict = binnacle.history
    train_loss=history_dict['loss'] #train loss
    val_loss=history_dict['val_loss'] #validation loss
    epoch=range(1,len(train_loss)+1) #epocas


    plt.figure(figsize=(5,5)) #tamaño de la grafica
    plt.plot(epoch,train_loss,'--',label="train") #graficar los datos del train set
    plt.plot(epoch,val_loss,'--',label="validation") #graficar los datos del validation set
    plt.xlabel("epoch") #label x
    plt.ylabel("loss") #label y
    #mostrar y finalizar
    plt.legend()
    plt.savefig("./train_plot.jpg")
    plt.show()


if __name__=='__main__':
    run()