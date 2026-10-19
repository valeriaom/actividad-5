# Actividad 5 | Bank Marketing, comparación reproducible con MLflow

Proyecto académico original: clasificación de contratación de depósitos a plazo, con dataset oficial de [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Reproducción en Colab

1. Abrir `fuentes/entrena.ipynb` en Google Colab. Ejecutar las celdas de arriba abajo.
2. La descarga usa `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`; si UCI está temporalmente inaccesible, subir manualmente el CSV oficial a `datos/datos_ini/bank-additional-full.csv`.
3. El notebook instala MLflow y ejecuta `datos_prep.py` y `train.py` (aproximadamente varios minutos, variable por hardware).
4. Las salidas se guardan en `resultados/`; el backend de seguimiento MLflow es `mlflow.db` (SQLite), con artefactos locales. Descargar el ZIP al acabar; Colab es efímero.
5. Para ver las ejecuciones, puede ejecutarse localmente `mlflow ui --backend-store-uri sqlite:///mlflow.db`, ya que `localhost` de Colab no es visible desde el navegador del usuario. Como evidencia alternativa, el notebook presenta ejecuciones registradas mediante MlflowClient y exporta sus métricas.

## Reproducción local

```bash
python -m pip install -r requirements.txt
python fuentes/datos_prep.py  # este módulo es importable; la preparación la ejecuta el notebook
python fuentes/train.py --root .
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Antes de ejecutar `train.py` directamente, ejecutar la sección de preparación del notebook para generar el CSV limpio. No se inventan métricas ni capturas de MLflow.

## Decisiones técnicas

- Se usa `bank-additional-full.csv` (41 188 observaciones de campañas históricas, antes de limpieza); no constituye evidencia sobre clientes actuales.
- Se excluye `duration` por fuga de información: se conoce después de la llamada.
- `unknown` se conserva como categoría; `pdays=999` es una marca de ausencia de contacto previo, no una medida ordinal normal: se convierte a cadena.
- Se eliminan únicamente duplicados idénticos; no hay identificador único que permita deduplicación de personas con seguridad.
- Validación estratificada 5 folds dentro de train (80 %) y test aislado (20 %).
- Escalado, imputación y one-hot encoding se aprenden dentro de cada fold mediante Pipeline.
- La selección se realiza por ROC-AUC en validación cruzada; el test se usa únicamente para evaluación final.
- La comparación Wilcoxon entre folds es exploratoria por dependencia entre particiones y limitada potencia; no demuestra causalidad ni superioridad universal.
- La fuente está ordenada temporalmente; un split aleatorio estima generalización i.i.d., **no** valida generalización a campañas futuras. En producción se exigiría evaluación temporal y deriva de datos.

## Estructura

`datos/datos_ini`: fuente original; `datos/datos_limp`: copia normalizada; `fuentes/datos_prep.py`: preparación; `fuentes/train.py`: experimentos y logging; `fuentes/entrena.ipynb`: notebook documentado; `resultados`: gráficos y tablas después de ejecutar; `evidencias`: capturas genuinas, añadidas tras ejecutar MLflow.

## Versionamiento y trazabilidad

`CHANGELOG.md` documenta las versiones lógicas; usar commits Git y etiqueta de versión al publicar. Cada corrida registra SHA256 del CSV limpio y seed. Para comprobar procedencia se almacena el SHA256 original en el reporte de calidad.

## Referencias primarias

- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- [Google Cloud MLOps](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Compatibilidad con MLflow / skops
El script de entrenamiento registra los modelos con `skops` y confía expresamente en `numpy.dtype` y `sklearn.compose._column_transformer._RemainderColsList` para el pipeline generado por el propio proyecto. Solo cargar artefactos de procedencia confiable.

### Serialización skops
Al registrar modelos propios con MLflow se autorizaron explícitamente `numpy.dtype`, `sklearn.compose._column_transformer._RemainderColsList` y `sklearn.tree._tree.Tree`. Solo deben emplearse estos tipos de confianza para artefactos generados y controlados localmente; no cargar modelos de procedencia desconocida con autorizaciones indiscriminadas.
