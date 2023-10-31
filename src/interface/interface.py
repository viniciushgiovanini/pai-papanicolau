import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

#############################################
#                 Metodos                   #
#############################################  
def zoom_in(event):
    canvas.scale("all", 0, 0, 1.2, 1.2)

def zoom_out(event):
    canvas.scale("all", 0, 0, 0.8, 0.8)
    
    
def selecionar_imagem():
    global imagem
    global imagem_tk
    arquivo = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg")])
    imagem = Image.open(arquivo)
    imagem_tk = ImageTk.PhotoImage(imagem)
    
    canvas.configure(state="normal")
    canvas.create_image(0, 0, anchor="nw", image=imagem_tk)
    canvas.image = imagem_tk
   
    
    nome_arquivo = arquivo.split("/")
    
    nome_final_arquivo = nome_arquivo[len(nome_arquivo)-1]
    
    campo_input.insert(0, nome_final_arquivo)

   
    
#############################################
#                  MAIN                     #
#############################################


# Inicializar a janela
windows = tk.Tk()
windows.geometry("900x600")
windows.title("Análise do Exame de Papanicolau")

#Configuracao do Grid 
windows.grid_rowconfigure(0, weight=0)
windows.grid_columnconfigure(0, weight=1)

# Campo de entrada do arquivo da imagem
campo_input = tk.Entry(windows, width=60)
campo_input.grid(row=1, column=0, pady=10)

botao_enviar = tk.Button(windows, text="Selecionar Imagem", command=selecionar_imagem)
botao_enviar.grid(row=2, column=0, pady=10)

# Exibir imagem no canvas
canvas = tk.Canvas(windows, width=400, height=400, state="disabled")
canvas.grid(row=4,column=0, pady=10)

# Label para exebir a imagem
label_imagem = tk.Label(windows)
label_imagem.grid(row=6, column=0, pady=5)

# Iniciar o loop da interface
windows.mainloop()


