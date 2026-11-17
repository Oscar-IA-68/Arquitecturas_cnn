# Evaluación comparativa de arquitecturas convolucionales

Proyecto académico de la universidad **Hybridge** para la asignatura de Sistemas de Aprendizaje Profundo. El proyecto utiliza CIFAR-10 para construir, entrenar y comparar arquitecturas convolucionales mediante un notebook de Jupyter.

## Objetivos

- Diseñar e implementar dos arquitecturas CNN propias.
- Utilizar una arquitectura clásica mediante transfer learning.
- Aplicar data augmentation y regularización.
- Mantener un flujo reproducible para comparar las arquitecturas.
- Persistir los modelos, historiales, configuraciones y resultados.
- Analizar experimentalmente el comportamiento de cada modelo.

La conclusión final de la comparación debe escribirse en la sección **Conclusiones** del notebook.

## Contenido del proyecto

```text
Arquitecturas_cnn/
├── Arquitecturas_cnn.ipynb
├── requirements.txt
├── README.md
└── artefactos/
    ├── modelos/
    ├── historiales/
    ├── configuraciones/
    └── comparaciones/
```

La carpeta `artefactos/` está destinada a los archivos generados durante los experimentos:

- `modelos/`: modelos Keras guardados en formato `.keras`.
- `historiales/`: métricas de entrenamiento y validación en formato `.csv`.
- `configuraciones/`: semillas, hiperparámetros y parámetros del experimento en formato `.json`.
- `comparaciones/`: métricas finales de prueba en formato `.csv` y `.json`.

El dataset CIFAR-10 se descarga mediante `keras.datasets.cifar10.load_data()` y se administra usando la caché de Keras; no es necesario incluirlo dentro del repositorio.

## Flujo del notebook

1. Importar las dependencias y establecer la semilla global.
2. Activar las operaciones deterministas de TensorFlow.
3. Definir el data augmentation.
4. Construir las dos CNN didácticas:
   - CNN secuencial personalizada.
   - CNN residual personalizada.
5. Construir MobileNetV2 con transfer learning y sus pesos de ImageNet congelados.
6. Cargar CIFAR-10 y separar entrenamiento, validación y prueba.
7. Redimensionar las imágenes de `32x32` a `64x64`.
8. Convertir las etiquetas a representación one-hot.
9. Entrenar una arquitectura seleccionada mediante `ARQUITECTURA`.
10. Guardar el modelo, el historial y la configuración.
11. Cargar los modelos disponibles y evaluarlos con el conjunto de prueba.
12. Generar los archivos de comparación y las gráficas.

La celda de entrenamiento debe ejecutarse una vez para cada valor de `ARQUITECTURA`:

```python
ARQUITECTURA = "vgg16"
```

También se pueden utilizar `"resnet50"` y `"mobilenet"`. Los nombres `vgg16` y `resnet50` corresponden a las dos CNN didácticas personalizadas del notebook; no representan las arquitecturas oficiales de Keras.

## Preprocesamiento y normalización

Las imágenes originales de CIFAR-10 llegan como valores RGB enteros en el intervalo `[0, 255]`. El pipeline las convierte a `float32`, las redimensiona a `64x64` y aplica el escalado dentro de cada modelo:

- CNN secuencial y CNN residual: `Rescaling(1./255)`, intervalo `[0, 1]`.
- MobileNetV2: `Rescaling(1./127.5, offset=-1)`, intervalo `[-1, 1]`, que es el preprocesamiento esperado por la red base.

El augmentation se aplica únicamente al flujo de entrenamiento. En validación y prueba las capas aleatorias funcionan en modo de inferencia.

## Reproducibilidad y comparación

Se utiliza la semilla `42` para Python, NumPy y TensorFlow/Keras. La semilla se restablece antes de crear cada experimento para controlar la inicialización de los modelos. Además, TensorFlow solicita operaciones deterministas.

La comparación utiliza la misma partición estratificada, los mismos parámetros generales y el mismo conjunto de prueba para las tres arquitecturas. Las métricas principales son `test_loss`, `test_accuracy` y el número de parámetros.

## Instalación

Con el entorno virtual activado, instalar las dependencias con:

```powershell
pip install -r requirements.txt
```

Después, abrir `Arquitecturas_cnn.ipynb` en Jupyter o Visual Studio Code y ejecutar las celdas en orden.

## Conclusiones

Esta sección queda reservada para documentar los resultados obtenidos, identificar la arquitectura con mejor desempeño y explicar las posibles razones de la diferencia observada.
