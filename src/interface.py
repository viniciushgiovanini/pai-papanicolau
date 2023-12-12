import os
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from AI.process import Process
from AI.segmentation import Segmentation
from AI.train_validation import TrainValidation
from keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

#############################################
#                 Metodos                   #
#############################################
class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')


class Zoom_Advanced(ttk.Frame):
    ''' Advanced zoom of the image '''
    
    def __init__(self, mainframe, path, imagem, resize):

        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=4, column=0, sticky='ns')
        hbar.grid(row=5, column=0, sticky='we')
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=4, column=0, sticky='nswe')
        # self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        
        self.master.rowconfigure(4, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        # with window and MacOS, but not Linux
        self.canvas.bind('<MouseWheel>', self.wheel)
        # only with Linux, wheel scroll down
        self.canvas.bind('<Button-5>',   self.wheel)
        # only with Linux, wheel scroll up
        self.canvas.bind('<Button-4>',   self.wheel)
        self.image = Image.open(path)  # open image
        self.image = imagem.resize(resize)
        self.width, self.height = self.image.size
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(
            0, 0, self.width, self.height, width=0)
        # Plot some optional random rectangles for the test purposes
        minsize, maxsize, number = 5, 20, 10
        # for n in range(number):
        #     x0 = random.randint(0, self.width - maxsize)
        #     y0 = random.randint(0, self.height - maxsize)
        #     x1 = x0 + random.randint(minsize, maxsize)
        #     y1 = y0 + random.randint(minsize, maxsize)
        #     color = ('red', 'orange', 'yellow', 'green', 'blue')[
        #         random.randint(0, 4)]
            # self.canvas.create_rectangle(
            #     x0, y0, x1, y1, fill=color, activefill='black')
        self.show_image()

    # Método para atualizar a imagem no canvas após demarcar os núcleos.
    def atualizar_imagem(self, imagem_nova):
      self.canvas.itemconfig(self.image, image=imagem_nova)
      self.canvas.update()
    
    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass  # Ok! Inside the image
        else:
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or window (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30:
                return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale:
                return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        # rescale all canvas objects
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()

    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        # get coordinates (x1,y1,x2,y2) of the image tile
        x1 = max(bbox2[0] - bbox1[0], 0)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            # sometimes it is larger on 1 pixel...
            x = min(int(x2 / self.imscale), self.width)
            # ...and sometimes not
            y = min(int(y2 / self.imscale), self.height)
            image = self.image.crop(
                (int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(
                image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk


class UInterface(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.label_resultado = None
        self.modelBinario = load_model(os.getcwd() + "/AI/notebook/model/binario/modelo_treinado_teste_100_sem_dropout.h5")
        self.modelCategorical = load_model(os.getcwd() + "/AI/notebook/model/categorical/modelo_treinado_teste_categorical_200.h5")
        self.parent = parent
        self.imagem = None
        self.arquivo = None
        self.initUI()
        self.dict_crescimento_regiao = None

    def initUI(self):
        self.parent.title("Análise do Exame de Papanicolau")

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        # Botão de Escolha de Imagem.
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Selecionar Imagem", command=lambda: self.selecionar_imagem(self.parent))
        menubar.add_cascade(label="Arquivo", menu=fileMenu)

        # Botão de Expandir Nucleos.
        menubar.add_command(label="Expandir Núcleos", command=lambda: self.expandir_nucleos(self.parent))
        segmentarMenu = Menu(menubar)
        segmentarMenu.add_command(label="Segmentação Por Região", command=lambda: self.regioes())
        segmentarMenu.add_command(label="Segmentação Por Equalização", command=lambda: self.equailizacao())
        menubar.add_cascade(label="Segmentação", menu=segmentarMenu)

        # Botão de scatterplot.
        scat = Menu(menubar)
        scat.add_command(label="Scatterplot Binario", command=lambda: Process.genScatterplotCompBin(self, "./AI/csv_pt2_binario.csv", self.arquivo))
        scat.add_command(label="Scatterplot Categorical", command=lambda: Process.genScatterplotCompCat(self, "./AI/csv_pt2_categorical.csv", self.arquivo))
        menubar.add_cascade(label="Gerar Scatterplot", menu=scat)
        # Adicionar um campo de entrada de texto
        self.entry = Entry(self.parent)
        self.entry.grid(row=0, column=0, padx=380, pady=10)
        
    def verificarValue(self):
        entry_value = self.entry.get()
        return int(entry_value) if entry_value else 100
    
    def equailizacao(self):
      obj = Segmentation(self.verificarValue())
      ret_dict_img, dict_recortada = obj.segmentacaoEqualizacao(self.arquivo)
      
      objProcess = Process(self.verificarValue())
      ret_distancias = objProcess.distanciaCentros(ret_dict_img)
      self.viewSegmentadas(ret_dict_img, ret_distancias, dict_recortada)
    
    def regioes(self):
        obj = Segmentation(self.verificarValue())
        ret_dict_img, dict_recortada = obj.segmentacaoRegiao(self.arquivo)
        
        objProcess = Process(self.verificarValue())
        ret_distancias = objProcess.distanciaCentros(ret_dict_img)
        self.viewSegmentadas(ret_dict_img, ret_distancias, dict_recortada)

    # Função para exibir resultados Resnet Bin
    def viewResnetBin(self, img_recortada_value, image_window):
        result, value = TrainValidation.classificarResnet(img_recortada_value, True, self.modelBinario)
        
        t = f"Resultado Resnet Bin: {result}, Valor: {value}"

        if self.label_resultado == None:
          self.label_resultado = tk.Label(image_window, text=t)
          self.label_resultado.grid(row=8, column=0, sticky="ns", padx=5)
        else:
          self.label_resultado.configure(text=t)
          
        image_window.update()

    def viewResnetCategorical(self, img_recortada_value, image_window):
        result, value = TrainValidation.classificarResnet(img_recortada_value, False, self.modelCategorical)
        
        t = f"Resultado Resnet Categorical: {result}, Valor: {value}"
        
        if self.label_resultado == None:
          self.label_resultado = tk.Label(image_window, text=t)
          self.label_resultado.grid(row=8, column=0, sticky="ns", padx=5)
        else:
          self.label_resultado.configure(text=t)
          
        image_window.update()

    def viewMahanalobisBin(self, img_recortada_value, image_window):
        train_validation_instance = TrainValidation()
        objProcess = Process(self.verificarValue())
        imgInCV2 = objProcess.convertPILtoCV2(img_recortada_value)
        predicao = train_validation_instance.classificarMahalanobis(imgInCV2, "./AI/csv_pt2_binario.csv", self.arquivo)
        
        t = f"Predição Mahanalobis Binário: {predicao}"

        if self.label_resultado == None:
          self.label_resultado = tk.Label(image_window, text=t)
          self.label_resultado.grid(row=8, column=0, sticky="ns", padx=5)
        else:
          self.label_resultado.configure(text=t)
        
        image_window.update()
        
    def viewMahanalobisCategorical(self, img_recortada_value, image_window):
        train_validation_instance = TrainValidation()
        objProcess = Process(self.verificarValue())
        imgInCV2 = objProcess.convertPILtoCV2(img_recortada_value)
        predicao = train_validation_instance.classificarMahalanobis(imgInCV2, "./AI/csv_pt2_categorical.csv", self.arquivo)
        
        t = f"Predição Mahanalobis Categorical: {predicao}"

        if self.label_resultado == None:
          self.label_resultado = tk.Label(image_window, text=t)
          self.label_resultado.grid(row=8, column=0, sticky="ns", padx=5)
        else:
          self.label_resultado.configure(text=t)
          
        image_window.update()

    def viewSegmentadas(self, dict_img_view, dict_distancia, dict_recortada):
        canvas_dois = tk.Canvas(self.parent)
        canvas_dois.grid(row=6, column=0, sticky="nsew")

        frame_dois = tk.Frame(canvas_dois)
        canvas_dois.create_window((0, 0), window=frame_dois, anchor="nw")

        scrollbar = tk.Scrollbar(self.parent, command=canvas_dois.yview, width=15, takefocus=True)
        scrollbar.grid(row=6, column=1, sticky="ns", padx=5)

        self.parent.grid_rowconfigure(7, weight=0)

        row = 0
        col = 0
        col_max = 5

        for cell_id, img in dict_img_view.items():
            imagem_pil = img.resize((150, 150))
            imagem_tk2 = ImageTk.PhotoImage(imagem_pil)
            label_dois = tk.Label(frame_dois, image=imagem_tk2)
            label_dois.imagem = imagem_tk2
            label_dois.grid(row=row+1, column=col, padx=5)

            # Cria um Label para exibir o nome da imagem
            label_nome = tk.Label(frame_dois, text=cell_id)
            label_nome.grid(row=row, column=col, padx=5)

                       
            # Vincular o clique da imagem à função de clique
            label_dois.bind("<Button-1>", lambda event, cell_id=cell_id: self.on_image_click(cell_id, dict_distancia, dict_recortada))

            col += 1
            if col == col_max:
                col = 0
                row += 2

    # Função para abrir uma nova janela com a imagem clicada
    def open_image_window(self, img, distancia, cell_id, img_recortada_value):
        image_window = tk.Toplevel()
        image_window.title("Visualização")

        width = image_window.winfo_screenwidth()
        height = image_window.winfo_screenheight()
        pos_x = ((width - 600) // 2)
        pos_y = ((height - 600) // 2)
        
        image_window.geometry(f"600x600+{pos_x}+{pos_y}")

        menubar = Menu(image_window)
        image_window.config(menu=menubar)

        self.label_resultado = None
        
        # Botão de Classificações.
        classificationMenu = Menu(menubar)
        classificationMenu.add_command(label="Mahalanobis Binário", command=lambda: self.viewMahanalobisBin(img, image_window))
        classificationMenu.add_command(label="Mahalanobis Categórico", command=lambda: self.viewMahanalobisCategorical(img, image_window))
        classificationMenu.add_command(label="CNN Resnet Binária", command=lambda: self.viewResnetBin(img_recortada_value, image_window))
        classificationMenu.add_command(label="CNN Resnet Categórico", command=lambda: self.viewResnetCategorical(img_recortada_value, image_window))
        menubar.add_cascade(label="Classificação", menu=classificationMenu)
        
        obj = Zoom_Advanced(image_window, self.arquivo, imagem=img, resize=(600, 600))

        label_nome = tk.Label(image_window, text=f"Distância da celula com ID: {cell_id} em px: {distancia}")
        label_nome.grid(row=7, column=0, sticky="ns", padx=5)

    # Função para lidar com o clique na imagem
    def on_image_click(self, cell_id, dict_distancia, dict_recortada):
        cell_info = dict_distancia.get(cell_id)
        img = cell_info.get("imagem")
        distancia = cell_info.get("distancia")
        
        img_recortada_value = dict_recortada.get(cell_id)
        
        self.open_image_window(img, distancia, cell_id, img_recortada_value)
            
    # Botão para selecionar a imagem para visualização com zoom.
    def selecionar_imagem(self, mainframe):
        self.arquivo = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png;*.jpg")])
        self.imagem = Image.open(self.arquivo)

        Zoom_Advanced(mainframe, path=self.arquivo, imagem=self.imagem, resize=(900, 600))

    # Botão para expandir os núcleos da imagem que foi selecionada.
    def expandir_nucleos(self, mainframe):
        obj = Process(self.verificarValue())
        nova_img = obj.markNucImage(self.arquivo)
        obj = Zoom_Advanced(mainframe, self.arquivo, imagem=nova_img, resize=(900, 600))
        obj.atualizar_imagem(nova_img)

######################
#        MAIN        #
###################### 
def main():
    # Inicializar a janela
    window = Tk()
    UInterface(window)
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    pos_x = ((width - 900) // 2)
    pos_y = ((height - 900) // 2)
    
    window.geometry(f"900x900+{pos_x}+{pos_y}")

    window.mainloop()

if __name__ == '__main__':
    main()