# Se encarga de comparar un rostro detectado con los registrados en la BD,
# usando la distancia euclidiana entre los landmarks

import numpy as np
from src.db.database import get_all_faces

def compare_faces(new_landmarks, threshold=0.6):
    """
    Compara un rostro nuevo con los almacenados en la BD.
    Retorna el nombre de la persona o en su caso 'Desconocido'.
    """
    faces = get_all_faces()
    if not faces:
        return "Sin registros", 0.0

    # Convertimos los nuevos landmarks en un vector plano (1D)
    new_vec = np.array(new_landmarks).flatten()
    best_name, best_dist =  "Desconocido", float("inf")


    for name, landmarks_str in faces:
        try:
            db_vec = np.array(eval(landmarks_str)).flatten()
        except Exception as e:
            print(f"Error Comparando con {name}: {e}")
            continue

        # Aseguramos que ambos tengan el mismo tamaño
        if len(new_vec) != len(db_vec):
            continue
        dist = np.linalg.norm(new_vec - db_vec)

        if dist < best_dist:
            best_name, best_dist = name, dist

    # Si la mejor distancia está dentro del umbral, se considera coincidencia
    if best_dist < threshold:
        return best_name, best_dist
    return "Desconocido", best_dist