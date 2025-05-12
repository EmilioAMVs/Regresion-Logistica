# Predicción de Diabetes con Regresión Logística

Este proyecto implementa un modelo de Regresión Logística utilizando el dataset de la base de datos Pima Indians Diabetes. Se busca predecir si una paciente padece o no de diabetes en base a características clínicas y personales.

## 📊 Dataset

**Fuente:** National Institute of Diabetes and Digestive and Kidney Diseases  
**Variables:**
- `Pregnancies`: Número de embarazos
- `Glucose`: Concentración de glucosa en plasma
- `BloodPressure`: Presión sanguínea diastólica (mm Hg)
- `SkinThickness`: Espesor del pliegue cutáneo del tríceps (mm)
- `Insulin`: Nivel de insulina sérica (mu U/ml)
- `BMI`: Índice de masa corporal (peso en kg / (altura en m)^2)
- `DiabetesPedigreeFunction`: Función de pedigree de diabetes
- `Age`: Edad
- `Outcome`: 0 (No diabético), 1 (Diabético)

Todas las pacientes son mujeres mayores de 21 años de ascendencia Pima.

## 🧠 Modelo

Se ha implementado un modelo de **Regresión Logística** con los siguientes pasos:
- División del dataset en conjuntos de entrenamiento y prueba (75% - 25%)
- Escalado de características con `StandardScaler`
- Entrenamiento del modelo
- Evaluación con:
  - Matriz de confusión
  - Precisión
  - Visualización de fronteras de decisión (Glucosa y BMI)

## 📈 Resultados

- **Predicción para Glucosa=120 y BMI=30.5**: No diabético
- **Matriz de Confusión:**
```
[[116 14]
[[29 33]]
```
- **Precisión del modelo:** 78%

## 📉 Visualizaciones

Las siguientes gráficas muestran las regiones de decisión del modelo para el conjunto de entrenamiento y de prueba utilizando las características más significativas: **Glucosa** y **BMI**.

![Entrenamiento](resultados_entrenamiento.png)

![Prueba](resultados_prueba.png)


---

### 🔍 Análisis de Resultados

#### 📌 Interpretación del Modelo
- Se ha obtenido una **precisión del 78%**, lo cual es aceptable para un modelo base sin ajustes avanzados.
- La matriz de confusión indica:
  - **116 verdaderos negativos**: pacientes correctamente clasificados como no diabéticos.
  - **33 verdaderos positivos**: pacientes correctamente clasificados como diabéticos.
  - **14 falsos positivos**: clasificó incorrectamente como diabéticos.
  - **29 falsos negativos**: diabéticos no detectados por el modelo.

#### 🎯 Visualización
- La **frontera de decisión** muestra que el modelo aprende una división lineal basada en la Glucosa y el BMI.
- Aunque la mayoría de los puntos están correctamente clasificados, hay zonas de confusión especialmente en el rango medio de glucosa.

#### 💡 Conclusiones
- **Glucosa** y **BMI** son variables clave y suficientes para una visualización 2D efectiva.
- Podría mejorarse el rendimiento:
  - Añadiendo regularización o ajustando hiperparámetros.
  - Aplicando técnicas de selección de características o modelos más complejos.
- El modelo es interpretable y útil como primer paso para cribado en salud pública.

---

