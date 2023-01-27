import tensorflow as tf
import logging

def run():
    logging.info("creando modelo")

    model=create()
    compile(model)
    model.save('./models/created_model.h5')

    logging.info("modelo guardado exitosamente")

def create():
    logging.info("creando capas del modelo")

    model=tf.keras.Sequential([

        #crear capas convolucionales
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4),input_shape=(150,100,3)),
        tf.keras.layers.MaxPool2D(2,2),#tamaño de la matriz
        tf.keras.layers.BatchNormalization(), #normalizacion de los datos de 0-1

        tf.keras.layers.Dropout(0.3), #dropout para conbatir el sobreentrenamiento

        tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

        #capa flatten para aplastar a 1D
        tf.keras.layers.Flatten(),

        #capa densa
        tf.keras.layers.Dense(units=100,activation='swish'),

        #capa de salida
        tf.keras.layers.Dense(units=1,activation='sigmoid')
    ])  
    model.summary()
    logging.info("modelo creado")

    return model

def compile(model):
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logging.info("modelo compilado")
    
if __name__=='__main__':
    run()