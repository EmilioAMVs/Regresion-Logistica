# Regresión Logística (Dataset de Diabetes con Glucose y BMI para visualización)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargar el dataset
dataset = pd.read_csv('diabetes.csv')

# Solo para visualización 2D usamos dos características (Glucose y BMI)
X = dataset[['Glucose', 'BMI']].values
y = dataset['Outcome'].values

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado de características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicción de un nuevo paciente (Ejemplo)
nuevo = classifier.predict(sc.transform([[120, 30.5]]))
print(f"Predicción para Glucose=120 y BMI=30.5: {'Diabético' if nuevo[0]==1 else 'No diabético'}")

# Predicción en conjunto de prueba
y_pred = classifier.predict(X_test)

# Matriz de Confusión y Precisión
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Matriz de Confusión:\n", cm)
print(f"Precisión del modelo: {accuracy:.2f}")

# ----------------------------
# Visualización - Conjunto de Entrenamiento
# ----------------------------
X_set, y_set = sc.inverse_transform(X_train), y_train  # Visualización en escala original
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Regresión Logística (Conjunto de Entrenamiento)')
plt.xlabel('Glucosa')
plt.ylabel('BMI')
plt.legend()
plt.show()

# ----------------------------
# Visualización - Conjunto de Prueba
# ----------------------------
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Regresión Logística (Conjunto de Prueba)')
plt.xlabel('Glucosa')
plt.ylabel('BMI')
plt.legend()
plt.show()
