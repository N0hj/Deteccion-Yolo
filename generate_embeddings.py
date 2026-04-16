"""
generate_embeddings.py
======================
Lee las fotos de los estudiantes en `students_photos/`, extrae los embeddings
faciales con FaceNet (InceptionResnetV1) y guarda un archivo `embeddings.pkl`.

Ejecutar UNA VEZ (o cuando se agreguen nuevos estudiantes):
    python generate_embeddings.py

Formato de nombres de archivo soportado:
    Carlos_1.jpg   Carlos_2.jpg   →  nombre = "Carlos"
    Maria1.jpeg    Maria2.jpeg    →  nombre = "Maria"
    Juan_foto_1.png              →  nombre = "Juan_foto"
"""

import os
import sys
import pickle
import re
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Agrega el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STUDENTS_FOLDER, EMBEDDINGS_FILE


# ─── Utilidades ──────────────────────────────────────────────────────────────

def extract_name(filename: str) -> str:
    """
    Extrae el nombre del estudiante desde el nombre de archivo.
    Ejemplos:
        'Carlos_1.jpg'   → 'Carlos'
        'Maria2.jpeg'    → 'Maria'
        'Juan_foto_1.png'→ 'Juan_foto'
    """
    stem = Path(filename).stem            # 'Carlos_1'
    # Intenta quitar el último segmento si es puramente numérico
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()
    # Quita dígitos finales
    name = re.sub(r"\d+$", "", stem).strip("_").strip()
    return name if name else stem


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── Función principal ────────────────────────────────────────────────────────

def generate_embeddings(students_folder: str, output_file: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Dispositivo: {device}")
    print(f"📂  Carpeta de fotos: {students_folder}\n")

    if not os.path.isdir(students_folder):
        print(f"❌  No existe la carpeta '{students_folder}'.")
        print("    Créala y agrega las fotos de los estudiantes.")
        sys.exit(1)

    # Inicializar modelos
    mtcnn  = MTCNN(image_size=160, margin=20, min_face_size=40,
                   thresholds=[0.6, 0.7, 0.7], device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Acumular embeddings por nombre
    raw: dict[str, list[np.ndarray]] = {}

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in os.listdir(students_folder)
             if Path(f).suffix.lower() in supported]

    if not files:
        print("⚠️  No se encontraron imágenes en la carpeta.")
        sys.exit(1)

    print(f"{'Archivo':<35} {'Nombre detectado':<25} {'Estado'}")
    print("─" * 70)

    for filename in sorted(files):
        name    = extract_name(filename)
        img_path = os.path.join(students_folder, filename)

        try:
            img          = Image.open(img_path).convert("RGB")
            face_tensor  = mtcnn(img)          # → Tensor[3,160,160] o None

            if face_tensor is None:
                print(f"  {filename:<33} {name:<25} ⚠️  Sin cara detectada")
                continue

            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0).to(device))
                emb = emb.cpu().numpy()[0]      # shape (512,)

            raw.setdefault(name, []).append(emb)
            print(f"  {filename:<33} {name:<25} ✅")

        except Exception as exc:
            print(f"  {filename:<33} {name:<25} ❌  {exc}")

    if not raw:
        print("\n❌  No se procesó ninguna imagen correctamente.")
        sys.exit(1)

    # ── Construir representación final por estudiante ──────────────────────
    # Guardamos TODOS los embeddings individuales (no solo el promedio)
    # para mejorar la precisión con pocas fotos.
    final: dict[str, np.ndarray] = {}
    print("\n📊  Resumen por estudiante:")
    for name, embs in sorted(raw.items()):
        arr = np.stack(embs)          # (N, 512)
        # Normalizar individualmente y luego promediar
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        final[name] = arr             # guardamos todas las fotos
        print(f"   • {name}: {len(embs)} foto(s)")

    with open(output_file, "wb") as f:
        pickle.dump(final, f)

    print(f"\n✅  Embeddings guardados en: {output_file}")
    print(f"📚  Estudiantes registrados: {sorted(final.keys())}\n")
    return final


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_embeddings(STUDENTS_FOLDER, EMBEDDINGS_FILE)
