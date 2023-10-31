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
    self.value_expand = 50
    self.img_analisada = None
    
    
  def limparDiretorio(self, path):
    for arquivo in os.listdir(path):
      caminho_arquivo = os.path.join(path, arquivo)
      if os.path.isfile(caminho_arquivo):
          os.remove(caminho_arquivo)
          
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
        
  def markNucImage(self, path_image):
    df = pd.read_csv("./data/classifications.csv")
    
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
          cv2.imwrite(f'{path_preview}{nome_img_selecionada}', img_marcada)
          
          # Limpar a variavel cv2
          cv2.destroyAllWindows()        
      

######################
#   MAIN PROVISORIO  #
###################### 
if __name__ == "__main__":
 
 obj = Process()
 obj.markNucImage("D:\AREA_DE_TRABALHO\Faculdade_6_Periodo\pai\pai-papanicolau\src\AI\data/dataset/1c900ddde4d55e63c0d06c4854b29f89.png")
      

