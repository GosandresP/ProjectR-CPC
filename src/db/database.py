# Crear una BD


import sqlite3
import os


BD_PATH = "data/face_data.db"

# Función de incialización de la base de datos

def init_db():
    """Crear la Base de datos y la tabla si no existe"""
    os.makedirs("data", exist_ok=True)
    # Estructura de la tabla
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            landmarks NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Def insert_face para insertar datos en la tabla
def insert_face(name, landmarks):
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, landmarks) VALUES (?, ?)", (name, str(landmarks)))
    conn.commit()
    conn.close()

# Def get_faces para obtener datos de la tabla
def get_all_faces():
    conn = sqlite3.connect(BD_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, landmarks FROM faces")
    faces = cursor.fetchall()
    conn.close()
    return faces


