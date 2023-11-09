import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Arreglos de datos
x = np.array([[10,0], [0,-10], [5,-2], [5,10], [0,5], [5,5]])
y = np.array([0,0,0,1,1,1])

#Dividir los datos en conjuntos de entrenamiento y pruebas
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size = 0.2, random_state=42)

#Crear un clasificador KNN
k=3 #Número de Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
#Entrenar el clasificador
knn_classifier.fit(x_train, y_train)

#Clasificamos un nuevo dato
dato = [[2,1]]
prediccion = knn_classifier.predict(dato)
print("El dato se clasificó en: ", prediccion)

#Separar los puntos por clases
x_clase0 = x[y==0]
x_clase1 = x[y==1]

#Crear un nuevo grafico
plt.figure (figsize = (11, 11))

#Graficar los puntos de las clases (Clase 0: Azul, clase 1: Rojo)
plt.scatter (x_clase0[:,0],x_clase0[:,1],c="blue",label="Clase 0")

plt.scatter (x_clase1[:,0],x_clase1[:,1],c="red",label="Clase 0")

#Graficamos el dato nuevo
plt.scatter(2,1,c="yellow", label="Nuevo dato")

#Agrega etiquetas de ejes
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.legend()

#Muestra el grafico
plt.grid(True)
plt.show()