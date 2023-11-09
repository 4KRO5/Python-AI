import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar los datos desde el archivo CSV
dataset = pd.read_csv("DatosTM.csv")

# Supongamos que tienes una columna "x" y una columna "Peso"
X = dataset[['Diametro', 'Peso']].values
y = dataset['Etiqueta'].values

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el clasificador SVM
svm_classifier = SVC(kernel='linear')  # Usamos un kernel lineal

# Entrenamos el modelo SVM
svm_classifier.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
predictions = svm_classifier.predict(X_test)

# Calculamos la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del modelo SVM: {accuracy}")

# Imprimimos el número de datos de prueba
print(f"El número de datos de prueba es de {len(X_test)}")

# Separar los datos por clases
clase0 = X[y == 0]
clase1 = X[y == 1]
clase2 = X[y == 2]
clase3 = X[y == 3]

# Crear una gráfica de dispersión (scatter plot)
plt.figure(figsize=(8, 8))
plt.scatter(clase0[:, 0], clase1[:, 1], label='Canelitas', color='blue', market='o')
plt.scatter(clase1[:, 0], clase1[:, 1], label='Galletas', color='yellow', market='o')
plt.scatter(clase2[:, 0], clase2[:, 1], label='Crackers', color='red', market='o')
plt.scatter(clase3[:, 0], clase3[:, 1], label='Coco', color='green', market='o')

# Añadir etiquetas y título a la gráfica
plt.xlabel('Diámetro')
plt.ylabel('Peso')
plt.title('Gráfica de Datos de Galletas')

# Añadir leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()