# Detector de Casas Colombianas — YOLOv8

Proyecto de detección de objetos entrenado para identificar casas en imágenes colombianas usando YOLOv8. Incluye API REST con FastAPI para inferencia en tiempo real.

---

##  Ejemplo de detección

![Detección de ejemplo](static/ejemplo.jpg)

> Detección con confianza de 0.84 sobre imagen de casas colombianas coloniales.

---

## Resultados del modelo

| Métrica       | Valor  |
|---------------|--------|
| **mAP@0.5**   | 0.623  |
| **Precision** | 0.668  |
| **Recall**    | 0.607  |
| **Épocas**    | 64/100 (EarlyStopping) |
| **Modelo base** | YOLOv8n |

---

## Dataset

- **Total imágenes:** 194 (155 entrenamiento (80%) / 39 validación(20%))
- **Origen:** Google Street View + fotografías de casas colombianas de google
- **Variedad:** urbano, rural, distintos, regiones, ciudades, pueblos, estratos y ángulos. 
- **Anotación:** Roboflow — clase única `house`
- ** Dataset en Google Drive:** [Ver dataset](https://drive.google.com/drive/folders/1XCH-w6WQ-oVb_cCxFO66bMlNWp9ylA5n?usp=sharing)

---

## Estructura del repositorio

```
taller-yolo-deteccion-casas-Florez-Mendoza-Rodriguez/
├── src/
│   ├── train_yolo.py       # Script de entrenamiento
│   └── inferencia.py       # Script de inferencia por terminal
├── static/
│   └── index.html          # Interfaz visual
├── models/
│   └── house-yolo.pt       # Pesos del modelo entrenado
├── data/
│   ├── train/
│   │   ├── images/         # Imágenes de entrenamiento
│   │   └── labels/         # Anotaciones .txt (formato YOLO)
│   └── valid/
│       ├── images/         # Imágenes de validación
│       └── labels/
├── app.py                  # API FastAPI
├── data.yaml               # Configuración del dataset
├── requirements.txt        # Dependencias
└── README.md
```

---

## Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/MeCaSifred/Taller-yolo-deteccion-casas-Florez-Mendoza-Rodriguez.git
cd Taller-yolo-deteccion-casas-Florez-Mendoza-Rodriguez

# 2. Crea y activa el entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Instala las dependencias
pip install -r requirements.txt
```

---

## Reproducir entrenamiento

1. Descarga el dataset desde Google Drive y colócalo en `data/`
2. Ajusta las rutas absolutas en `data.yaml`
3. Ejecuta:

```bash
python src/train_yolo.py
```

Los pesos finales se guardarán automáticamente en `models/house-yolo.pt`.

**Hiperparámetros usados:**

| Parámetro | Valor |
|-----------|-------|
| epochs | 100 |
| imgsz | 640 |  
| batch | 8 |
| optimizer | AdamW |
| lr0 | 0.001 |
| patience | 20 |
| mosaic | 1.0 |
| flipud | 0.0 |
| fliplr | 0.5 |

---

## Inferencia por terminal

```bash
python src/inferencia.py ruta/a/imagen.jpg
```

---

## API FastAPI

### Iniciar el servidor

```bash
python app.py
```

Abre en el navegador:
- **http://localhost:8000/docs** → Interfaz interactiva para probar la API
- **http://localhost:8000/health** → Estado del modelo

### Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Página principal |
| POST | `/detectar` | Recibe imagen → retorna JSON con detecciones |
| POST | `/detectar-imagen` | Recibe imagen → retorna imagen con bounding boxes |
| GET | `/health` | Estado de la API y modelo |

### Ejemplo de respuesta `/detectar`

```json
{
  "total_casas_detectadas": 1,
  "detecciones": [
    {
      "confianza": 0.84,
      "bbox": [430.5, 62.3, 1087.2, 681.4]
    }
  ],
  "imagen_resultado": "/static/resultado_abc123.jpg",
  "mensaje": " Detección completada"
}
```

---

## Limitaciones y pasos futuros

- **Dataset pequeño** (39 imágenes validación) puede causar sobreajuste
- **Recall de 0.607** indica que algunas casas no se detectan — mejoraría con más imágenes
- **Recomendaciones:**
  - Ampliar dataset a 500+ imágenes
  - Probar modelo `yolov8s.pt` (más capacidad)
  - Agregar más clases: apartamento, edificio, finca campesina


---

## Autores

- Andrés Florez - David Rodriguez - Sifred Mendoza  
- Curso: Aplicaciones de Aprendizaje Automático de Máquinas
