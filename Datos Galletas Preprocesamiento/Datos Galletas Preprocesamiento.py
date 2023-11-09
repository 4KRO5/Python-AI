import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo .csv
dataset = pd.read_csv('Datos Galletas Preprocesamiento/Datos_TV.csv')  # Reemplaza 'ruta_completa' con la ubicación real del archivo CSV

# Datos
x = dataset[['Diametro', 'Peso']].values
y = dataset['Etiqueta'].values

# Separar los datos por clase
clase0 = x[y == 0]
clase1 = x[y == 1]
clase2 = x[y == 2]
clase3 = x[y == 3]

# Crear la gráfica
plt.figure(figsize=(8, 8))  # Opcional: ajustar el tamaño de la gráfica
plt.scatter(clase0[:, 0], clase0[:, 1], label="Marias", color="blue", marker="o")
plt.scatter(clase1[:, 0], clase1[:, 1], label="Coco", color="yellow", marker="o")
plt.scatter(clase2[:, 0], clase2[:, 1], label="Crackets", color="red", marker="o")
plt.scatter(clase3[:, 0], clase3[:, 1], label="Canelitas", color="green", marker="o")
plt.xlabel("Diametro")
plt.ylabel("Peso")
plt.title("GRÁFICA DE GALLETAS")

plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()