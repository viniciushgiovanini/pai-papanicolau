import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import os
from AI.process import Process
import warnings

warnings.filterwarnings("ignore")

class Segmentation():
  '''
    Classe que tem os metodos para segmentação das imagens
    
  '''
  
  def __init__(self, value_expand):
    self.obj = Process(value_expand)
  
  
  
  def crescimentoRegiao(self, image, seed, threshold):
    '''
      Função para realização do crescimento por região
      
      Parâmetros:
        image (cv2): imagem para segmentar
        seed (tupla): posição central da imagem
        threshold (int): limiar de corte
      
      Return:
        image (np.array): imagem segmentada em array
    '''    
    
    
    # Pega a altura e a largura
    altura, largura = image.shape
    # Inicia matriz para pegar os px visitados
    matriz_visitados = np.zeros((altura, largura), dtype=np.uint8)
    # Matriz para pegar região segmentada
    matriz_segmentado = np.zeros_like(image)
    
    # Inicia a lista com a semente
    list_px = []
    list_px.append(seed)

    while list_px:
        # Retira um px da lista
        x, y = list_px.pop()
        # Verifica se o px foi visitado
        if not matriz_visitados[x, y]:
            # Verifica se o px esta dentro do limiar de similiadade com a semente
            # if np.linalg.norm(image[x, y] - image[seed]) < threshold:
            if np.linalg.norm(image[x, y].astype(float) - image[seed].astype(float)) < threshold:
                # Marca o px como visitado
                matriz_visitados[x, y] = 1
                # Adiciona o px na regiao segmentada
                matriz_segmentado[x, y] = image[x, y]
                # Percorre os px vizinhos do px
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        # Verifica se o vizinho esta dentro dos limites da imagem.
                        if 0 <= x + dx < altura and 0 <= y + dy < largura:
                            # Add vizinho na lista
                            list_px.append((x + dx, y + dy))

    return matriz_segmentado
  
  
  
  
  def pegarPixelClaroeEscuroRegiao(self, imagem):

    # Encontrar o valor do pixel mais claro e mais escuro
    pixel_mais_escuro = np.min(imagem)
    pixel_mais_claro = np.max(imagem)
    
    return pixel_mais_escuro, pixel_mais_claro
  
  def segmentacaoRegiao(self, path_img):
    
    '''
      Função que faz o pré-processamento e pós usando o método de crescimento por região
      
      Parâmetros:
        path_img (String): Path da imagem a ser segmentada.
        
      Return:
        img_segmentation_list (list de PILLOW): Lista de nucleos img segmentados
    '''
    
    
    img_cut_dict = self.obj.cutNucImage(path_image=path_img)
  
    img_segmentation_dict = {}
    img_recortada = {}
    
    for key, image in img_cut_dict.items():
      
      altura, largura, _ = image.shape
      
      img_tratada = cv2.GaussianBlur(image, (7,7), 7)

      x = largura//2
      y = altura//2

      
      cv_image = cv2.cvtColor(img_tratada, cv2.COLOR_BGR2GRAY)
      
               
      px_escuro, px_claro = self.pegarPixelClaroeEscuroRegiao(cv_image)
      
      # Pega o threshold com a soma do pixel mais escuro com o mais claro e divide por 2, 
      # caso seja mt baixo o valor menor que 40 ele soma os valores
      threshold = (int(px_claro) - int(px_escuro)) / 4
        
     
      region = self.crescimentoRegiao(cv_image, (x,y), threshold)
      
      contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      mask = np.zeros_like(region)

      cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

      result_image = cv2.bitwise_and(image, image, mask=mask)
      
      img_convert_toPIL = Image.fromarray(result_image)
      
      img_convert_img_normal_toPIL = Image.fromarray(image)
      
      img_segmentation_dict[key] = img_convert_toPIL.copy()
      img_recortada[key] = img_convert_img_normal_toPIL.copy()
     
     
    return img_segmentation_dict, img_recortada
  
  
  def segmentacaoEqualizacao(self, path_img):
    
    '''
      Função que faz o pré-processamento e pós usando filtro gausiano, equalizacao do histograma, filtro de realce e mascara invertida
      
      Parâmetros:
        path_img (String): Path da imagem a ser segmentada.
        
      Return:
        img_segmentada_list (list de PILLOW): Lista de nucleos img segmentados
    '''
    
    img_cut_dict = self.obj.cutNucImage(path_image=path_img)
    
    img_segmentada_dict = {}
    
    img_recortada_dict = {}
    
    for key, image in img_cut_dict.items():
    
      img_tratada = cv2.GaussianBlur(image, (7,7), 7)

      # Converte a imagem para o espaço de cor HSV
      imagem_hsv = cv2.cvtColor(img_tratada, cv2.COLOR_BGR2HSV)

      # Separa os canais HSV
      h, s, v = cv2.split(imagem_hsv)

      # Equaliza o canal de valor
      v_equalizado = cv2.equalizeHist(v)

      # Combina os canais novamente
      imagem_hsv_equalizada = cv2.merge([h, s, v_equalizado])

      # Converte a imagem de volta para o espaço de cor original
      imagem_equalizada = cv2.cvtColor(imagem_hsv_equalizada, cv2.COLOR_HSV2BGR)

      # Ajustando a luminosidade (brilho) da imagem equalizada
      alpha = 9.0  # Fator de brilho
      beta = 40  # Valor de contraste
      adjusted_image = cv2.convertScaleAbs(imagem_equalizada, alpha=alpha, beta=beta)
      
      gray_original = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

      altura, largura, _ = adjusted_image.shape
      
      x = largura//2
      y = altura//2
      
      
      
      # Defina um valor de limiar para separar a região escura
      px_escuro, px_claro = self.pegarPixelClaroeEscuroRegiao(gray_original)
      
      threshold_value = (int(px_claro)+int(px_escuro))
      
      
      while(threshold_value >= 200):
        threshold_value -= 100
      
      # Crie uma máscara com base no limiar
      _, dark_mask = cv2.threshold(gray_original, threshold_value, 255, cv2.THRESH_BINARY)

      # Invertendo a mascara
      dark_mask = cv2.bitwise_not(dark_mask)


      # Aplique a máscara na outra imagem colorida
      result_image = cv2.bitwise_and(image, image, mask=dark_mask)
      img_convert_toPIL = Image.fromarray(result_image)
      
      img_convert_toPIL_recortada = Image.fromarray(image)
      
      img_segmentada_dict[key] = img_convert_toPIL.copy()
      img_recortada_dict[key] = img_convert_toPIL_recortada.copy()
    return img_segmentada_dict, img_recortada_dict
  

# if __name__ == '__main__':
#   obj = Segmentation(70)
#   ret = obj.segmentacaoRegiao('D:\AREA_DE_TRABALHO\Faculdade_6_Periodo\pai\pai-papanicolau\src\AI\data\dataset/363b6b00d925e5c52694b8f7b678c53b.png')

#   print(len(ret))
  
#   for key,each in ret.items():
#     print(f"ID {key}, imagem {str(each)}")
#     each.show()