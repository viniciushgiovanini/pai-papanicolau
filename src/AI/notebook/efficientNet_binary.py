import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import time

tf.config.list_physical_devices()
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
        '../data/segmentation_dataset_binario_treino/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',color_mode='rgb')

test_dataset = test_datagen.flow_from_directory(
        '../data/segmentation_dataset_binario_treino/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',color_mode='rgb')



# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding layers on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    # Dropout(0.5), 
    Dense(1, activation='sigmoid')  # Binary classification, so use sigmoid activation
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.F1Score(threshold=0.5)])

# Exiba um resumo do modelo
start_time = time.time()

epochs = 20

resultados = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

end_time = time.time()

model.summary()

tempo_de_treinamento = (end_time - start_time) / 60

print("#######################################")
print("Tempo de treinamento:", tempo_de_treinamento, "minutos")
print("#######################################")

plt.plot(resultados.history["f1_score"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["F1-Score"])
plt.savefig("f1_score_binary.png")
plt.clf()


plt.plot(resultados.history["loss"])
plt.plot(resultados.history["val_loss"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Erro treino", "Erro teste"])
plt.savefig("graph/loss_binary.png")
plt.clf()



plt.plot(resultados.history["binary_accuracy"])
plt.plot(resultados.history["val_binary_accuracy"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Acuracia Binaria treino", "Acuracia Binaria teste"])
plt.savefig("graph/acuracia_binary.png")
plt.clf()


plt.plot(resultados.history["accuracy"])
plt.plot(resultados.history["val_accuracy"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Acuracia treino", "Acuracia teste"])
plt.savefig("graph/acuracia.png")
plt.clf()

model.save(f"model/modelo_treinado_teste_{str(epochs)}.h5")
