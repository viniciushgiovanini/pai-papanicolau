if __name__ == '__main__':
  obj = Segmentation(70)
  ret = obj.segmentacaoHistograma('./data/dataset/363b6b00d925e5c52694b8f7b678c53b.png')

  print(len(ret))
  
  for each in ret:
    each.show()