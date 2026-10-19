# Actividad 5 · Bank Marketing y MLflow

Proyecto académico de clasificación bancaria. Se comparan regresión logística, Random Forest y SVM lineal mediante validación cruzada estratificada, GridSearchCV y seguimiento de experimentos con MLflow.

**Fuente de datos:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

**Estado:** repositorio de la Actividad 5. Los resultados experimentales y las evidencias se conservan en el respaldo académico; se incorporarán los archivos fuente sin alterar las métricas ya registradas.

## Reproducción

El archivo `fuentes/entrena.ipynb` (incluido en el paquete académico de la actividad) se ejecuta en Google Colab. Se excluye `duration` de los predictores porque su valor no está disponible antes del contacto.

Los modelos se evalúan mediante accuracy, precision, recall, F1 y ROC-AUC. La comparación estadística se interpreta sin afirmar significación cuando el valor p supera 0.05.

## Autoría

Valeria Ochoa Mondragón · Maestría en Inteligencia Artificial Aplicada.
