import os
import sys
import time
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    YOLO_MODEL, YOLO_PERSON_CLASS, YOLO_CONF_THRESH,
    EMBEDDINGS_FILE, RECOGNITION_THRESHOLD,
    DOOR_OPEN_DURATION, CAMERA_INDEX,
)

GREEN = (34, 197, 94)
RED   = (0,  60, 220)
WHITE = (255, 255, 255)

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# carga model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

print("Cargando YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Cargando FaceNet...")
mtcnn  = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# embeddings
if not os.path.isfile(EMBEDDINGS_FILE):
    print(f"ERROR: No se encontró '{EMBEDDINGS_FILE}'.")
    print("Ejecuta primero: python generate_embeddings.py")
    sys.exit(1)

with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings = pickle.load(f)

print(f"Estudiantes cargados: {sorted(embeddings.keys())}")

#funciones
def recognize(pil_img):
    """Retorna (nombre, confianza) si reconoce, o (None, 0) si no."""
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None, 0.0

    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device))
        emb = emb.cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    best_name = None
    best_dist = float("inf")

    for name, known_embs in embeddings.items():
        for k_emb in known_embs:
            dist = cosine_distance(emb, k_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist < RECOGNITION_THRESHOLD:
        return best_name, 1.0 - best_dist

    return None, 0.0
#loop principal
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"ERROR: No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
    sys.exit(1)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H     = 70         
door_until   = 0.0        
last_name    = ""

print("Sistema activo. Presiona [q] para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    now  = time.time()

    # deteccion yolo 
    results = yolo(frame, classes=[YOLO_PERSON_CLASS],
                   conf=YOLO_CONF_THRESH, verbose=False)[0]

    person_detected   = len(results.boxes) > 0
    recognized_person = None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #recorte
        crop     = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        name, conf = recognize(pil_crop)

        if name:
            recognized_person = name
            door_until        = now + DOOR_OPEN_DURATION
            last_name         = name
            box_color         = GREEN
            label             = f"{name}  {conf:.0%}"
        else:
            box_color = RED
            label     = "Desconocido"

        # bouding box y etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 13, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6), FONT, 0.65, WHITE, 2)


    door_open = now < door_until

    if door_open:
        banner_color = GREEN
        banner_text  = f"[ PUERTA ABIERTA ]  —  {last_name}"
    elif person_detected and not recognized_person:
        banner_color = RED
        banner_text  = "[ ACCESO DENEGADO ]  —  Persona no reconocida"
    else:
        banner_color = (45, 45, 45)
        banner_text  = "[ ESCANEANDO ]"

    cv2.rectangle(frame, (0, 0), (w, BANNER_H), banner_color, -1)
    cv2.putText(frame, banner_text, (20, 45), FONT, 0.85, WHITE, 2)
    cv2.imshow("Control de Acceso", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistema detenido.")