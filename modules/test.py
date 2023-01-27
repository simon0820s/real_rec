import tensorflow as tf
import get_data
import logging

logging.basicConfig(level=logging.INFO)

def run():
    logging.info("[INFO] cargando data")

    x_test,y_test=get_data.run(train=False)#funcion de obtener datos
    model=tf.keras.models.load_model('./models/trained_model.h5')#funcion de cargar modelo

    logging.info("[INFO] haciendo testing")

    score=model.evaluate(x_test,y_test)#guardando el puntaje del set de validacion
    print('Precisión en el set de validación: {:.1f}%'.format(100*score[1])) #print de el rendimiento en la funcion de perdida en porcentaje
    print(f"score: {score}") #score en consola
    
    logging.info("[INFO] modelo testeado")
    
if __name__=='__main__':
    run()