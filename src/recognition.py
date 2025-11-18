# Se encarga de comparar un rostro detectado con los registrados en la BD,
# usando la distancia euclidiana entre los landmarks

import numpy as np
from collections import deque
from typing import Deque, Tuple
from src.db.database import get_all_faces

_EMA_STATE: Deque[float] = deque(maxlen=1)

def _normalize_landmarks(landmarks):
    arr = np.array(landmarks)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return (arr - mean) / std

def _flatten_landmarks(landmarks):
    return np.array(landmarks).flatten()

def _compute_ema(value, alpha):
    if not _EMA_STATE:
        _EMA_STATE.append(value)
        return value
    smoothed = alpha * value + (1 - alpha) * _EMA_STATE[-1]
    _EMA_STATE.append(smoothed)
    return smoothed

def compare_faces(new_landmarks, threshold=0.6, ema_alpha=0.4, use_normalization=True):
    """
    Compara un rostro nuevo con los almacenados en la BD y aplica suavizado EMA
    para reducir saltos en la confianza.
    """
    faces = get_all_faces()
    if not faces:
        return "Sin registros", 0.0, 0.0

    processed = _normalize_landmarks(new_landmarks) if use_normalization else np.array(new_landmarks)
    new_vec = _flatten_landmarks(processed)
    new_norm = np.linalg.norm(new_vec)
    if new_norm == 0:
        return "Desconocido", 1.0, 0.0
    new_unit = new_vec / new_norm
    best_name, best_sim = "Desconocido", 0.0

    for name, landmarks_str, _ in faces:
        try:
            stored = np.array(eval(landmarks_str))
            stored = _normalize_landmarks(stored) if use_normalization else stored
            db_vec = _flatten_landmarks(stored)
            db_norm = np.linalg.norm(db_vec)
            if db_norm == 0:
                continue
            db_unit = db_vec / db_norm
        except Exception as e:
            print(f"Error Comparando con {name}: {e}")
            continue
        if len(new_unit) != len(db_unit):
            continue
        sim = np.dot(new_unit, db_unit)
        if sim > best_sim:
            best_name, best_sim = name, sim

    confidence = best_sim
    smoothed_conf = _compute_ema(confidence, ema_alpha)
    if best_sim > threshold:
        return best_name, 1.0 - best_sim, smoothed_conf
    return "Desconocido", 1.0 - best_sim, smoothed_conf

def reset_ema_state():
    _EMA_STATE.clear()

def get_last_confidence():
    return _EMA_STATE[-1] if _EMA_STATE else 0.0
