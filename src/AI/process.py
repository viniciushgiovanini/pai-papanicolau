import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import shutil

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
      img_marcada = cv2.putText(img_marcada, str(img_id), (posi_x-10, posi_y-55) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2) 
    
    # Converter cv2 para PIL -> a classe de zoom usa demonstração em PIL.
    img_toPIL = Image.fromarray(img_marcada)

    return img_toPIL
  
  
  def cutNucImage(self,path_image):
    if not (isinstance(self.value_expand, int)):
         self.value_expand = 100

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
      
      
      x1 = posi_x - self.value_expand
      y1 = posi_y - self.value_expand
      x2 = posi_x + self.value_expand
      y2 = posi_y + self.value_expand
      
      # Recortando a imagem;
      img_recortada = img[y1:y2,x1:x2]
        
      
      if(len(img_recortada)!= 0): 
        img_cut_dict[cell_id] = img_recortada.copy()
      
      # Limpar a variavel cv2
      cv2.destroyAllWindows() 
    return img_cut_dict
  

