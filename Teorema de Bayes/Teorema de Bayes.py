import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Cargar los datos desde el archivo CSV
dataset = pd.read_csv("archivo.csv")

# Separar características (X) y etiquetas (y)
X = dataset.drop('clase', axis=1)  # Asumiendo que la columna objetivo se llama 'clase'
y = dataset['clase'].values

# Dividir los datos en un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador Bayesiano
clf = GaussianNB()

# Entrenar el clasificador con los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión de las predicciones
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy * 100:.2f}%")

# Separar los datos por clase
clase0 = X_test[y_test == 0]
clase1 = X_test[y_test == 1]
clase2 = X_test[y_test == 2]
clase3 = X_test[y_test == 3]

# Configurar la gráfica
plt.figure(figsize=(8, 8))
plt.scatter(clase0['diametro'], clase0['peso'], label='Marias', marker='o')
plt.scatter(clase1['diametro'], clase1['peso'], label='Coco', marker='o')
plt.scatter(clase2['diametro'], clase2['peso'], label='Crackers', marker='o', color='red')
plt.scatter(clase3['diametro'], clase3['peso'], label='Canelitas', marker='o')

# Configurar etiquetas y título de la gráfica
plt.xlabel('Diámetro')
plt.ylabel('Peso')
plt.title('Gráfica de las galletas')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()