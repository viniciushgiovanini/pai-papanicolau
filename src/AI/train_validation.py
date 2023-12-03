import math
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
from scipy.spatial.distance import mahalanobis
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings("ignore")

class TrainValidation:
    # Resnet Binário.
    def classificarResnet(img, isBinario, model):
        # model = load_model(os.getcwd() + "/AI/notebook/model/binario/modelo_treinado_teste_100.h5")
        
        img = img.resize((100,100))
        # plt.imshow(img)
        # plt.show()
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        value = model.predict(images)
        # print(value[0])
        value = value[0].tolist()
        
        value_list_int = list(map(int, value))         
        
        maxValue = max(value_list_int)        
        valu_posi = value_list_int.index(maxValue)
        
        if(isBinario):
            if value[0] < 0.5:
                return "Negativo", value[0]
            else:
                return "Positivo", value[0]
        else:
            dictClassifier = {"ASC-H": 0, "ASC-US" : 1, "HSIL": 2, "LSIL": 3, "Negative": 4, "SCC": 5}
            for i,each in enumerate(dictClassifier):
            
                if(valu_posi == i):
                    return each, value[0]
                
    # Mahanalobis Binário.
    def calculaAreaPerimetroImagem(self, img_cv2):
  
      imagem_cinza = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
      _, mascara_binaria = cv2.threshold(imagem_cinza, 1, 255, cv2.THRESH_BINARY)
      contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      mascara_colorida = np.zeros_like(img_cv2)
      cv2.drawContours(mascara_colorida, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)

      area_branca = np.sum(mascara_binaria == 255)
      
      
      total_perimetro = 0
      
      for contorno in contornos:
          perimeter = cv2.arcLength(contorno, closed=True)
          total_perimetro += perimeter
      
      return area_branca, round(total_perimetro, 2)
    
    # 4 PI area / perimetro^2
    def calcularCompacidade(self, img_cv2):
        area, perimetro = self.calculaAreaPerimetroImagem(img_cv2)
        
        compacidade = (area * (4 * math.pi)) / (perimetro**2)
        
        return round(compacidade, 4)
    
    # Calcular excentricidade
    # 1-menor^2 / 1-maior^2
    def calcularExcentricidades(self, img_cv2):
        imagem_cinza = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, mascara_binaria = cv2.threshold(imagem_cinza, 1, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maior_contorno = max(contornos, key=cv2.contourArea)

            
        if len(maior_contorno) >= 5:
            elipse = cv2.fitEllipse(maior_contorno)
            eixo_maior = max(elipse[1])
            eixo_menor = min(elipse[1])
        else:
            return 0
        
        excentricidade = 1 - ((eixo_menor ** 2) / (eixo_maior ** 2))

        return excentricidade
    
    def calcular_estatisticas_por_classe(self, df):
        estatisticas_por_classe = {}
        for classe in df['label'].unique():
            classe_df = df[df['label'] == classe]
            
            # Converta as colunas para números
            numeric_columns = ['area', 'compacidade', 'excentricidade']
            for col in numeric_columns:
                classe_df[col] = pd.to_numeric(classe_df[col], errors='coerce')
            
            estatisticas_por_classe[classe] = {
                'media': np.mean(classe_df[numeric_columns], axis=0),
                # 'covariancia':  np.cov(classe_df[numeric_columns], rowvar=False)
                'covariancia': LedoitWolf().fit(classe_df[numeric_columns]).covariance_
            }

        return estatisticas_por_classe
    
    def gerarEstatisticas(self, path_csv, nomeArq):
        df = pd.read_csv(path_csv)
        
        splitPath = nomeArq.split("/")
        nomeImg = splitPath[len(splitPath)-1]
        
        dfs = df[df["image_filename"] == nomeImg]
        
        estatisticas_treinamento = self.calcular_estatisticas_por_classe(dfs)

        return estatisticas_treinamento
    
    def classificar_mahalanobis(self, amostra, estatisticas_por_classe):
        distancias = {}
        
        for classe, estatisticas in estatisticas_por_classe.items():   
          
            # calcular matrix inversa
            try:
              mat_inv = np.linalg.inv( estatisticas['covariancia'])
            except:
              mat_inv = np.linalg.inv( estatisticas['covariancia'])

            distancias[classe] = mahalanobis(amostra, estatisticas['media'], mat_inv)
        

        return min(distancias, key=distancias.get)
    
    def classificarMahalanobis(self, img, csvPath, nomeArq):


        area, _ = self.calculaAreaPerimetroImagem(img)
        compacidade = self.calcularCompacidade(img)
        excentricidade = self.calcularExcentricidades(img)

        amostra = np.array([area, compacidade, excentricidade])
        
        predicao = self.classificar_mahalanobis(amostra, self.gerarEstatisticas(os.getcwd() + csvPath, nomeArq))
        
        return predicao
      
      
# train_validation_instance = TrainValidation()
# img = cv2.imread(os.getcwd() + "/AI/data/segmentation_dataset_binario/Negativo/0a2a5a681410054941cc56f51eb8fbda.png-5636.png")

# print(os.getcwd() + "/AI/data/segmentation_dataset_binario/Negativo/0a2a5a681410054941cc56f51eb8fbda.png-5636.png")

# train_validation_instance.classificarMahalanobis(img, "/AI/csv_pt2_binario.csv", "0a2a5a681410054941cc56f51eb8fbda.png")