import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import math

class Process():
  def __init__(self, n):
    self.value_expand = n

  def markNucImage(self, path_image):
    # Determinando valor 100 caso nao for passado o valor de N
    if not (isinstance(self.value_expand, int)):
      self.value_expand = 100
    
    nome_img_selecionada = path_image.split("/")
    nome_img_selecionada = nome_img_selecionada[(len(nome_img_selecionada)-1)]
    df = pd.read_csv(os.getcwd() + "/AI/data/classifications.csv")
    df = df[df['image_filename'] == nome_img_selecionada]
    
    # Leitura da imagem com padrão RGB para preservar cores para conversão para PIL.
    img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)

    for each in df.iterrows():
      
      posi_x = each[1]['nucleus_x']
      posi_y = each[1]['nucleus_y']  
      img_id = each[1]['cell_id']
      
      x1 = posi_x - self.value_expand
      y1 = posi_y - self.value_expand
      x2 = posi_x + self.value_expand
      y2 = posi_y + self.value_expand

      # Fazendo o quadrado na imagem
      img_marcada = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      
      # Escrevendo o indice na imagem
      img_marcada = cv2.putText(img_marcada, str(img_id), (posi_x-10, posi_y-5) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2) 
    
    # Converter cv2 para PIL -> a classe de zoom usa demonstração em PIL.
    img_toPIL = Image.fromarray(img_marcada)

    return img_toPIL
  
  
  def cutNucImage(self,path_image):
    nome_img_selecionada = path_image.split("/")
    nome_img_selecionada = nome_img_selecionada[(len(nome_img_selecionada)-1)]

    df = pd.read_csv(os.getcwd() + "/AI/data/classifications.csv")
    df = df[df['image_filename'] == nome_img_selecionada]

    # Leitura da imagem com padrão RGB para preservar cores para conversão para PIL.
    img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)

    img_cut_dict = {}

    for each in df.iterrows():
      
      posi_x = each[1]['nucleus_x']
      posi_y = each[1]['nucleus_y']
      cell_id = each[1]["cell_id"]
      
      # Normalizar o tamanho do quadrado para não pegar uma imagem completada com preto.
      x1 = max(0, posi_x - self.value_expand)
      y1 = max(0, posi_y - self.value_expand)
      x2 = min(img.shape[1], posi_x + self.value_expand)
      y2 = min(img.shape[0], posi_y + self.value_expand)
                  
      # Recortando a imagem;
      img_recortada = img[y1:y2,x1:x2]
      
      if(len(img_recortada)!= 0):
        img_cut_dict[cell_id] = img_recortada.copy()
      
      # Limpar a variavel cv2
      cv2.destroyAllWindows() 
    return img_cut_dict
  
  
  def calcular_distancia(self, x1, y1, x2, y2):
    distancia = abs(round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),2))
    return distancia
  
  
  def convertPILtoCV2(self, img):
   return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  
  # def convertCV2toPIL(self, img):
  #   return Image.fromarray(img)
  
  def distanciaCentros(self, dict_img):
    
    '''
      Funcao que calcula a distancia entre os centros da imagem.
      
      Parametros: 
        dict_img (dict): dicionario com id e img em PIL
      
      return: 
        img_dist_dict (dict): id do nucleo com value sendo a distancia
    
    '''
    
    img_dist_dict = {}
    
    for key, image in dict_img.items():
      
      imgCv2 = self.convertPILtoCV2(image)
      
      # Converte a img orignal segmentada em tons de cinza
      gray_original_dois = cv2.cvtColor(imgCv2, cv2.COLOR_BGR2GRAY)
      
      # Encontra o contorno da img com tons de cinza
      contours, _ = cv2.findContours(gray_original_dois, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Verifica se tem contornos na imagem
      if len(contours) > 0:

          contour = contours[0]
          
          # Calcula o momentos do contorno (area)          
          M = cv2.moments(contour)
          
          # Calcula as coordenadas do centro do contorno
          
          # M10 --> Soma das coordenadas x dos pixels do contorno
          # M01 --> Soma das coordenadas y dos pixels do contorno
          # M00 --> divido pela area do contorno
          
          cx = int(M["m10"] / M["m00"])
          cy = int(M["m01"] / M["m00"])
      
      # Pega o centro da img do csv
      altura, largura, _ = imgCv2.shape
      
      vermelho = (255, 255, 255)  
      verde = (0, 255, 0)
      
      # Pintando centro da imagem segmentada
      imgCv2[cy, cx] = vermelho 
      # Pintando centro a partir do csv
      imgCv2[largura//2, altura//2] = verde  
      
      # Calculando a distancia entre os dois pontos
      ret = self.calcular_distancia(cx, cy, largura//2,altura//2) 
      # add o valor da distancia em um dicionario com a key sendo o id da celular e o value sendo a distancia euclidiana
      img_dist_dict[key] = ret
    
    return img_dist_dict

