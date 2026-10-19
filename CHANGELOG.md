# Registro de cambios

## 1.0.0 - Estructura inicial
- Dataset oficial identificado y función de descarga reproducible.
- Validación de esquema y control de calidad del dataset.
- Transformación reproducible, exclusión documentada de `duration`, eliminación de filas exactamente duplicadas.

## 1.1.0 - Diseño experimental
- Pipelines de regresión logística, Random Forest y SVM lineal.
- GridSearchCV, validación cruzada estratificada y test independiente.

## 1.2.0 - Trazabilidad
- Registro de métricas, parámetros, SHA256, modelos y gráficas en MLflow.
- Comparación estadística exploratoria y artefactos CSV/JSON.

Las entradas describen componentes preparados; la ejecución y publicación en GitHub deben comprobarse por separado.

## Corrección de compatibilidad MLflow
- Sustituido el backend de seguimiento de archivos por SQLite (`mlflow.db`) para compatibilidad con MLflow reciente.
- Actualizados notebook, auditoría de runs y guía de reproducción.

## Corrección de serialización MLflow (skops)
- Autorizados de forma explícita `numpy.dtype` y `sklearn.compose._column_transformer._RemainderColsList` en la persistencia de modelos propios con MLflow.
- Se mantiene la serialización `skops` y la base de seguimiento SQLite.

## Corrección de serialización de Random Forest
- Autorizado `sklearn.tree._tree.Tree` exclusivamente para los modelos entrenados localmente y registrados mediante `skops`.
- Se evita habilitar globalmente tipos no confiables.
