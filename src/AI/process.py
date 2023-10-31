import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import cv2
import tensorflow as tf
import os
import shutil


class Process():
  def __init__(self):
    self.diretorio_dataset = '../AI/data/dataset/'
    self.diretorio_dataset_recortado = "./data/dataset_recortado/"
    self.diretorio_dataset_marcado = "./data/dataset_marcado/"
    self.value_expand = 50
    self.img_analisada = None
        
  def setExpandValue(self, value_expand):
   self.value_expand = value_expand
   
  def setImgAnalisada(self, img_analisada):
   self.img_analisada = img_analisada
  
  def cutDataset(self):
    df = pd.read_csv("./data/filter_classifications.csv")

    for each in df.iterrows():
      
      nome_img = each[1]['image_filename']
      nome_da_doenca = each[1]['bethesda_system']
      posi_x = each[1]['nucleus_x']
      posi_y = each[1]['nucleus_y']  
      
      path_imagem_dataset_original = f'{self.diretorio_dataset}{nome_img}'

      # Verifica se a iamgem existe
      if(os.path.isfile(path_imagem_dataset_original)):
        # Onde ele vai ler cada imagem;
        img = cv2.imread(path_imagem_dataset_original)
        
        x1 = posi_x - self.value_expand
        y1 = posi_y - self.value_expand
        x2 = posi_x + self.value_expand
        y2 = posi_y + self.value_expand
        
        # Recortando a imagem;
        img_recortada = img[posi_y-self.value_expand:posi_y+self.value_expand, posi_x-self.value_expand:posi_x+self.value_expand]
        
        # Verifica se existe um folder no destino com o nome da doenca;
        if not os.path.exists(os.path.join(self.diretorio_dataset_recortado, nome_da_doenca)):
          os.mkdir(os.path.join(self.diretorio_dataset_recortado, nome_da_doenca))
          
        # Salva a imagem recortada no novo destino
        cv2.imwrite(f'{self.diretorio_dataset_recortado}{nome_da_doenca}/' + f'{nome_da_doenca}_{posi_x}_{posi_y}_{nome_img}', img_recortada)
        
        # Limpar a variavel cv2
        cv2.destroyAllWindows()
        
  def markNucImage(self):
    df = pd.read_csv("./data/filter_classifications.csv")

    nome_img_selecionada = self.img_analisada

    if(os.path.isfile(f'./data/dataset/{nome_img_selecionada}')):
      shutil.copy(f'{self.diretorio_dataset}{nome_img_selecionada}', f'{self.diretorio_dataset_marcado}{nome_img_selecionada}')

      for each in df.iterrows():
        
        nome_img = each[1]['image_filename']
        nome_da_doenca = each[1]['bethesda_system']
        posi_x = each[1]['nucleus_x']
        posi_y = each[1]['nucleus_y']  
        img_id = each[1]['cell_id']
        
        if(nome_img == nome_img_selecionada):
          
        
          path_imagem_dataset_original = f'{self.diretorio_dataset_marcado}{nome_img_selecionada}'

          # Verifica se a iamgem existe
          if(os.path.isfile(path_imagem_dataset_original)):
            # Onde ele vai ler cada imagem;
            img = cv2.imread(path_imagem_dataset_original)
            
            print(nome_da_doenca)
            print(nome_img)
            
            x1 = posi_x - self.value_expand
            y1 = posi_y - self.value_expand
            x2 = posi_x + self.value_expand
            y2 = posi_y + self.value_expand

            # Fazendo o quadrado na imagem
            img_marcada = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Escrevendo o indice na imagem
            img_marcada = cv2.putText(img_marcada, str(img_id), (posi_x-10, posi_y-55) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2) 
              
            # Salva imagem com marcacao na propria imagem
            cv2.imwrite(f'{self.diretorio_dataset_marcado}{nome_img_selecionada}', img_marcada)
            
            # Limpar a variavel cv2
            cv2.destroyAllWindows()        
        
        
        
        
