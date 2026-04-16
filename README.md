#  Sistema de Control de Acceso — YOLO + FaceNet

Reconocimiento facial en tiempo real para control de acceso de una puerta.  
Detecta personas con **YOLOv8**, identifica quién es con **FaceNet (InceptionResnetV1)**
y muestra un **banner verde** cuando se concede acceso.

---

---

## Instalación

### Crear entorno virtual 
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

---


1. Crea la carpeta `students_photos/` en el directorio del proyecto.
2. Agrega las fotos con este formato de nombre:

   ```
   NombreEstudiante_1.jpg
   NombreEstudiante_2.jpg
   ```


##  Uso

### Generar embeddings 
```bash
python generate_embeddings.py
```

### Ejecutar 
```bash
python main.py
```
