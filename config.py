import os

# --- Rutas ---
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
STUDENTS_FOLDER   = os.path.join(BASE_DIR, "students_photos")   # carpeta con fotos
EMBEDDINGS_FILE   = os.path.join(BASE_DIR, "embeddings.pkl")    # base de datos generada

# --- Modelo YOLO ---

YOLO_MODEL        = "yolov8n.pt"
YOLO_PERSON_CLASS = 0           
YOLO_CONF_THRESH  = 0.5         

RECOGNITION_THRESHOLD = 0.60
DOOR_OPEN_DURATION = 3        

# --- Cámara ---
CAMERA_INDEX      = 0           
CAMERA_WIDTH      = 1280
CAMERA_HEIGHT     = 720

