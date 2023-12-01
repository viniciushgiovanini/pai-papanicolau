import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
import os

class TrainValidation:
    def classificar(img, isBinario):
        model = load_model(os.getcwd() + "/AI/notebook/model/modelo_treinado_teste_100.h5")
        
        img = img.resize((100,100))
        # plt.imshow(img)
        # plt.show()
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        value = model.predict(images)
        print(value[0])
        value = value[0].tolist()
        
        value_list_int = list(map(int, value))         
        
        maxValue = max(value_list_int)        
        valu_posi = value_list_int.index(maxValue)
        
        if(isBinario):
            if value[0] > 0.85:
                return "Negativo", value[0]
            else:
                return "Positivo", value[0]
        else:
            dictClassifier = {"ASC-H": 0, "ASC-US" : 1, "HSIL": 2, "LSIL": 3, "Negative": 4, "SCC": 5}
            for i,each in enumerate(dictClassifier):
            
                if(valu_posi == i):
                    return each, value[0]