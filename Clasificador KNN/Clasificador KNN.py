import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# arreglo de datos
x = np.array([[1, 2], [2, 3], [1, 3],[2,2], [3, 4], [3, 2], [4, 2],[3,3]])
y = np.array([0, 0, 0,0, 1, 1, 1,1])

#dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42)

#crear un clasificador KNN
k=3 #numero de vecinos a considerar
knn_classifier = KNeighborsClassifier(n_neighbors=k)

#entrenar el clasificador KNN
knn_classifier.fit(x_train, y_train)

#clasificamos un nuevo dato
dato=[[2,1]]
prediccion=knn_classifier.predict(dato)
print("El dato se clasifico en" , prediccion)

# separar los puntos por clases
x_clase0 = x[y == 0]
x_clase1 = x[y == 1]

# crear un nuevo grafico
plt.figure(figsize=(11, 11))

# graficar los puntos de la clase 0 en azul
plt.scatter(x_clase0[:, 0], x_clase0[:, 1], c="blue", label="Clase 0")

# graficar los puntos de la clase 1 en rojo
plt.scatter(x_clase1[:, 0], x_clase1[:, 1], c="red", label="Clase 1")

#graficamos el dato nuevo
plt.scatter(2,1,c="yellow", label="Dato Nuevo")

# agrega etiquetas de ejes
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.legend()

# muestra el grafico
plt.grid(True)
plt.show()