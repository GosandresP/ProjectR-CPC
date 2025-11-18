# Crear una BD


import sqlite3
import os
from datetime import datetime


BD_PATH = "data/face_data.db"

# Función de incialización de la base de datos

def _ensure_created_at(cursor):
    cursor.execute("PRAGMA table_info(faces)")
    columns = [row[1] for row in cursor.fetchall()]
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE faces ADD COLUMN created_at TEXT")

def init_db():
    """Crear la Base de datos y la tabla si no existe"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            landmarks NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    _ensure_created_at(cursor)
    conn.commit()
    conn.close()

# Def insert_face para insertar datos en la tabla
def insert_face(name, landmarks, created_at=None):
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    timestamp = created_at or datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO faces (name, landmarks, created_at) VALUES (?, ?, ?)",
        (name, str(landmarks), timestamp)
    )
    conn.commit()
    conn.close()

# Def get_faces para obtener datos de la tabla
def get_all_faces():
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, landmarks, created_at FROM faces")
    faces = cursor.fetchall()
    conn.close()
    return faces

def get_face_by_name(name):
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, landmarks, created_at FROM faces WHERE name = ?",
        (name,)
    )
    face = cursor.fetchone()
    conn.close()
    return face
