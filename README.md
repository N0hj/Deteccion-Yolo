# 🔐 Sistema de Control de Acceso — YOLO + FaceNet

Reconocimiento facial en tiempo real para control de acceso de una puerta.  
Detecta personas con **YOLOv8**, identifica quién es con **FaceNet (InceptionResnetV1)**
y muestra un **banner verde** cuando se concede acceso.

---

## 📁 Estructura del proyecto

```
access_control/
├── config.py                 # Parámetros del sistema
├── generate_embeddings.py    # Paso 1: generar base de datos facial
├── main.py                   # Paso 2: ejecutar sistema en tiempo real
├── requirements.txt
└── students_photos/          # TUS FOTOS aquí
    ├── Carlos_1.jpg
    ├── Carlos_2.jpg
    ├── Maria_1.jpg
    ├── Maria_2.jpg
    └── ...
```

---

## ⚡ Instalación

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

> **GPU NVIDIA (opcional pero recomendado):**  
> Instala PyTorch con CUDA antes del paso anterior:  
> https://pytorch.org/get-started/locally/

---

## 🖼️ Preparar fotos de estudiantes

1. Crea la carpeta `students_photos/` en el directorio del proyecto.
2. Agrega las fotos con este formato de nombre:

   ```
   NombreEstudiante_1.jpg
   NombreEstudiante_2.jpg
   ```

   Ejemplos válidos:
   | Archivo           | Nombre detectado |
   |-------------------|-----------------|
   | `Carlos_1.jpg`    | Carlos          |
   | `Carlos_2.jpeg`   | Carlos          |
   | `Maria1.png`      | Maria           |
   | `Juan_foto_1.jpg` | Juan_foto       |

3. Se recomienda **3–5 fotos por persona** con diferentes ángulos/iluminación.
4. Las fotos deben tener la **cara claramente visible** (no muy pequeña, sin oclusiones).

---

## 🚀 Uso

### Paso 1 — Generar embeddings (una sola vez)
```bash
python generate_embeddings.py
```
Esto crea el archivo `embeddings.pkl` con las huellas faciales de cada estudiante.  
**Repite este paso cada vez que agregues nuevos estudiantes.**

### Paso 2 — Ejecutar sistema en tiempo real
```bash
python main.py
```

### Controles durante ejecución
| Tecla | Acción |
|-------|--------|
| `q`   | Salir |
| `r`   | Recargar embeddings (sin reiniciar) |

---

## 🎨 Interfaz de cámara

```
┌─────────────────────────────────────────────────────────┐
│  [ PUERTA ABIERTA ]              Acceso concedido — Carlos  │  ← Banner VERDE
├─────────────────────────────────────────────────────────┤
│                                                         │
│          ┌──────────────────┐                           │
│          │ Carlos  94%      │  ← Caja verde + nombre    │
│          │                  │                           │
│          │   [persona]      │                           │
│          └──────────────────┘                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Estados del banner:**
- 🟢 **VERDE** `[ PUERTA ABIERTA ]` — Persona reconocida, acceso concedido
- 🔴 **ROJO** `[ ACCESO DENEGADO ]` — Persona detectada pero no reconocida
- ⚫ **GRIS** `[ ESCANEANDO ]` — Sin personas en frame

---

## ⚙️ Ajuste de parámetros (`config.py`)

| Parámetro              | Valor por defecto | Descripción |
|------------------------|:-----------------:|-------------|
| `RECOGNITION_THRESHOLD`| `0.60`            | Umbral de distancia coseno. Baja si no reconoce (→ 0.65), sube si hay falsos positivos (→ 0.50) |
| `DOOR_OPEN_DURATION`   | `3`               | Segundos que el banner verde permanece visible |
| `YOLO_CONF_THRESH`     | `0.5`             | Confianza mínima de detección YOLO |
| `CAMERA_INDEX`         | `0`               | Índice de cámara (cambia si tienes varias) |
| `YOLO_MODEL`           | `yolov8n.pt`      | Modelo YOLO: `n` (rápido) → `s` → `m` → `l` (preciso) |

---

## 🔧 Solución de problemas

| Problema | Solución |
|----------|----------|
| `No se encontró embeddings.pkl` | Ejecuta `python generate_embeddings.py` primero |
| `Sin cara detectada` en generate | La foto es muy pequeña o la cara está tapada; usa fotos más claras |
| Reconoce mal (falsos negativos) | Baja `RECOGNITION_THRESHOLD` a `0.65` en `config.py` |
| Falsos positivos | Sube `RECOGNITION_THRESHOLD` a `0.50` en `config.py` |
| Cámara no abre | Cambia `CAMERA_INDEX` a `1` o `2` en `config.py` |
| Lento en CPU | Usa `yolov8n.pt` (ya es el más rápido), o instala PyTorch con CUDA |

---

## 🏗️ Arquitectura del pipeline

```
Frame de cámara
      │
      ▼
┌─────────────┐
│   YOLOv8    │  Detecta SOLO clase "persona" (ignora todo lo demás)
└──────┬──────┘
       │  Recorte de cada persona detectada
       ▼
┌─────────────┐
│    MTCNN    │  Detecta y alinea la cara dentro del recorte
└──────┬──────┘
       │  Tensor de cara [3×160×160]
       ▼
┌───────────────────────┐
│  InceptionResnetV1    │  Extrae embedding facial (vector 512D)
│  (VGGFace2 pretrained)│
└──────────┬────────────┘
           │  Distancia coseno vs. embeddings guardados
           ▼
    ¿Distancia < umbral?
       ┌───┴───┐
      SÍ      NO
       │       │
   VERDE     ROJO
  (acceso)  (denegado)
```
