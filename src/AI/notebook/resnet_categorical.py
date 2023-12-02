import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

tf.config.list_physical_devices()
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
        '../data/cnn_treino_categorical/train',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',color_mode='rgb')

test_dataset = test_datagen.flow_from_directory(
        '../data/cnn_treino_categorical/test/',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',color_mode='rgb')



# Load the pre-trained EfficientNetB0 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding layers on top of the base model

model = Sequential()

model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(train_dataset.class_indices), activation='softmax'))


optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy' ,tf.keras.metrics.AUC(num_thresholds=3),  tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Exiba um resumo do modelo
start_time = time.time()

epochs = 200

checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_recall', verbose=1, save_best_only=True, mode='max')

resultados = model.fit(
    train_dataset,
    steps_per_epoch=140,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[checkpoint]
)

end_time = time.time()

model.summary()

tempo_de_treinamento = (end_time - start_time) / 60

print("#######################################")
print("Tempo de treinamento:", tempo_de_treinamento, "minutos")
print("#######################################")


########################################
#       PLOTAR MATRIX DE CONFUSAO      #
########################################

y_true = test_dataset.classes
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_mat = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(25, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=train_dataset.class_indices.keys(),
            yticklabels=train_dataset.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f"graph/resnet/categorical/matrizdeconfusao_{epochs}.png")
plt.close()


plt.plot(resultados.history["loss"])
plt.plot(resultados.history["val_loss"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Erro treino", "Erro teste"])
plt.savefig("graph/resnet/categorical/loss_categorical.png")
plt.clf()



plt.plot(resultados.history["accuracy"])
plt.plot(resultados.history["val_accuracy"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Acuracia treino", "Acuracia teste"])
plt.savefig("graph/resnet/categorical/acuracia_categorical.png")
plt.clf()

plt.plot(resultados.history["auc"])
plt.plot(resultados.history["val_auc"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["AUC treino", "AUC teste"])
plt.savefig("graph/resnet/categorical/auc_categorical.png")
plt.clf()

plt.plot(resultados.history["recall"])
plt.plot(resultados.history["val_recall"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Recall treino", "Recall teste"])
plt.savefig("graph/resnet/categorical/recall_categorical.png")
plt.clf()

plt.plot(resultados.history["precision"])
plt.plot(resultados.history["val_precision"])
plt.title("Histórico de Treinamento")
plt.ylabel("Função de Custo")
plt.xlabel("Épocas de treinamento")
plt.legend(["Precisao treino", "Precisao teste"])
plt.savefig("graph/resnet/categorical/precisao_categorical.png")
plt.clf()



model.save(f"model/modelo_treinado_teste_categorical_{str(epochs)}.h5")
