import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import cv2
import tensorflow as tf
import os
import shutil


class Process():
  def __init__(self, n):
    self.value_expand = n

    
  def limparDiretorio(self, path):
    for arquivo in os.listdir(path):
      caminho_arquivo = os.path.join(path, arquivo)
      if os.path.isfile(caminho_arquivo):
          os.remove(caminho_arquivo)
        
  def markNucImage(self, path_image):
    df = pd.read_csv(os.getcwd() + "/data/classifications.csv")
    
    # Determinando valor 100 caso nao for passado o valor de N
    if not (isinstance(self.value_expand, int)):
      self.value_expand = 100
      
    
    # Fazendo caminho de onde vai ser salvo as imagens analisadas
    path_preview = os.getcwd() + '/data/tmp_img_preview/'
    
    # Verifica se existe o folder, caso não exista você cria ele (img_preview)
    if not os.path.exists(path_preview):
        os.makedirs(path_preview)
    
        
    # Limpando este caminho inicialmente
    self.limparDiretorio(path_preview)
    
    
    nome_img_selecionada = path_image.split("/")
    nome_img_selecionada = nome_img_selecionada[(len(nome_img_selecionada)-1)]
    
    # Copiando a imagem para o diretorio
    shutil.copy(path_image, path_preview + nome_img_selecionada)
    
        
    if (os.path.isfile(os.getcwd() + f'/data/dataset/{nome_img_selecionada}')):
      

      for each in df.iterrows():
        
        nome_img = each[1]['image_filename']
        nome_da_doenca = each[1]['bethesda_system']
        posi_x = each[1]['nucleus_x']
        posi_y = each[1]['nucleus_y']  
        img_id = each[1]['cell_id']
        
        if(nome_img == nome_img_selecionada):
          
        
          path_imagem_dataset_original = f'{path_preview}{nome_img_selecionada}'

          # Onde ele vai ler cada imagem;
          img = cv2.imread(path_imagem_dataset_original)
          
          x1 = posi_x - self.value_expand
          y1 = posi_y - self.value_expand
          x2 = posi_x + self.value_expand
          y2 = posi_y + self.value_expand

          # Fazendo o quadrado na imagem
          img_marcada = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
          
          # Escrevendo o indice na imagem
          img_marcada = cv2.putText(img_marcada, str(img_id), (posi_x-10, posi_y-55) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2) 
            
          # Salva imagem com marcacao na propria imagem
          cv2.imwrite(f'{path_preview}{nome_img_selecionada}', img_marcada)
          
          # Limpar a variavel cv2
          cv2.destroyAllWindows()        
######################
#   MAIN PROVISORIO  #
###################### 
# if __name__ == "__main__":
 
#  obj = Process(50)
      

