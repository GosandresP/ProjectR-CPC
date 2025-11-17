# APP.PY

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

from src.db.database import init_db, insert_face, get_all_faces
from src.recognition import compare_faces
from src.utils import draw_similarity_bar

# Iniciliazar l DB
init_db()

# Configuracion basica de la app
st.set_page_config(page_title="Reconocimiento Facial con MediaPipe", layout="wide")
st.markdown("# Reconocimiento Facial con **MediaPipe** para detectar y reconocer rostros en tiempo real usando la camara")

# Sidebar
menu = st.sidebar.radio("Menu", ["Reconocimientos en vivo", "Registrar nuevos rostros", "Ver registros"])

# Iniciar modulos de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==================
# Opcion #1 -> Reconocimientos en vivo
# ==================

if menu == "Reconocimientos en vivo":
    st.subheader("Camara en tiempo real en vivo")

    run = st.checkbox("Iniciar Camara")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 3,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh:
        while run:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo acceder a la camara")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Construir bloque de malla facial

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    name, dist = compare_faces(landmarks, threshold=0.6)
                    confidence = max(0.0, min(1.0, 1.0 - dist))


                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1,circle_radius=1
                        )
                    )
                    # Barra de similitud
                    frame = draw_similarity_bar(frame, confidence, x=30, y=110, w=260, h=18)

                    # Mostrar texto
                    cv2.putText(frame, f"{name}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confianza:", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()


# ==================
# Opcion #2 -> Registrar
# ==================

elif menu == "Registrar nuevos rostros":
    st.subheader("Registrar nuevos rostros")
    name_input = st.text_input("Nombre de la persona:")
    register_button = st.button("Registrar rostro")

    if register_button and name_input:
        cap = cv2.VideoCapture(0)
        st.info("Capturando rostro...")

        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,

            ) as face_mesh:
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    first = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]
                    insert_face(name_input, landmarks)
                    cv2.imwrite(f"data/captures/{name_input}.jpg", frame)
                    st.success("Rostro registrado con exito")
                else:
                    st.error("No se detecto ningun rostro. Intenta de nuevo.")

# ==================
# Opcion #2 -> Registrar
# ==================

elif menu == "Ver registros":
    st.subheader("Rostros registrados en la base de datos")
    registros = get_all_faces()
    if not registros:
        st.warning("No hay registros")

    else:
        for name, _ in registros:
            st.write(f"**{name}**")
            try:
                st.image(f"data/captures/{name}.jpg", width=200)
            except:
                st.caption("No se pudo cargar la imagen")