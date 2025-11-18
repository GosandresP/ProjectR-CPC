# APP.PY

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

from src.db.database import init_db, insert_face, get_all_faces, get_face_by_name
from src.recognition import compare_faces, reset_ema_state
from src.utils import draw_similarity_bar, enhance_lighting, set_camera_resolution, auto_zoom_on_face, inject_theme

# Iniciliazar l DB
init_db()

# Configuracion basica de la app
st.set_page_config(page_title="Reconocimiento Facial con MediaPipe", layout="wide")
st.markdown("# Reconocimiento Facial con **MediaPipe** para detectar y reconocer rostros en tiempo real usando la camara")
st.markdown(inject_theme(), unsafe_allow_html=True)

# Sidebar
menu = st.sidebar.radio("Menu", ["Reconocimientos en vivo", "Registrar nuevos rostros", "Ver registros"])
with st.sidebar:
    st.markdown("## Ajustes de Precisión")
    threshold = st.slider("Threshold", min_value=0.2, max_value=1.0, value=0.6, step=0.05)
    ema_alpha = st.slider("Suavizado EMA", min_value=0.05, max_value=0.9, value=0.4, step=0.05)
    normalize = st.checkbox("Normalizar landmarks", value=True)
    st.markdown("## Modo Largo Alcance")
    long_range = st.checkbox("Activar modo largo alcance")
    apply_filter = st.checkbox("Mejorar iluminación/contraste")
    auto_zoom = st.checkbox("Zoom automático")

# Iniciar modulos de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==================
# Opcion #1 -> Reconocimientos en vivo
# ==================

if menu == "Reconocimientos en vivo":
    st.subheader("Camara en tiempo real en vivo")

    col_cam, col_info = st.columns([2, 1])
    with col_cam:
        run = st.toggle("Iniciar Camara")
        stop_btn = st.button("Detener/Refrescar", type="secondary")
        FRAME_WINDOW = st.image([])
    with col_info:
        st.markdown("### Último reconocimiento")
        info_placeholder = st.empty()

    if stop_btn:
        run = False
        reset_ema_state()

    cap = cv2.VideoCapture(0)
    if long_range:
        set_camera_resolution(cap, 1920, 1080)
    else:
        set_camera_resolution(cap, 1280, 720)

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
                st.error("No se pudo acceder a la camara")
                break

            frame = cv2.flip(frame, 1)
            frame = enhance_lighting(frame, apply_filter)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    if auto_zoom:
                        frame = auto_zoom_on_face(frame, face_landmarks)
                    name, dist, confidence = compare_faces(
                        landmarks,
                        threshold=threshold,
                        ema_alpha=ema_alpha,
                        use_normalization=normalize
                    )
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
                    frame = draw_similarity_bar(frame, confidence, x=30, y=110, w=260, h=18)
                    cv2.putText(frame, f"{name}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confianza:", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    face_record = get_face_by_name(name) if name not in ["Desconocido", "Sin registros"] else None
                    with info_placeholder.container():
                        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                        if face_record:
                            img_path = f"data/captures/{name}.jpg"
                            try:
                                st.image(img_path, use_column_width=True)
                            except:
                                st.markdown("<div class='no-photo'>Sin fotografía almacenada</div>", unsafe_allow_html=True)
                            created_at = face_record[2] or "Sin fecha"
                            st.markdown(f"**Nombre:** {name}")
                            st.markdown(f"**Registrado:** {created_at}")
                        else:
                            st.markdown("<div class='no-photo'>Sin coincidencia</div>", unsafe_allow_html=True)
                        st.markdown(f"**Threshold aplicado:** {threshold:.2f}")
                        st.markdown(f"**Confianza suavizada:** {confidence*100:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        reset_ema_state()


# ==================
# Opcion #2 -> Registrar
# ==================

elif menu == "Registrar nuevos rostros":
    st.subheader("Registrar nuevos rostros")
    name_input = st.text_input("Nombre de la persona:")
    register_button = st.button("Registrar rostro")

    if register_button and name_input:
        cap = cv2.VideoCapture(0)
        if long_range:
            set_camera_resolution(cap, 1920, 1080)
        st.info("Capturando rostro...")

        ret, frame = cap.read()

        if ret:
            frame = enhance_lighting(frame, apply_filter)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,

            ) as face_mesh:
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    first = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]
                    insert_face(name_input, landmarks, created_at=datetime.now().isoformat())
                    cv2.imwrite(f"data/captures/{name_input}.jpg", frame)
                    st.success("Rostro registrado con exito")
                else:
                    st.error("No se detecto ningun rostro. Intenta de nuevo.")
        cap.release()


# ==================
# Opcion #2 -> Registrar
# ==================

elif menu == "Ver registros":
    st.subheader("Rostros registrados en la base de datos")
    registros = get_all_faces()
    if not registros:
        st.warning("No hay registros")

    else:
        cols = st.columns(3)
        for idx, (name, _, created_at) in enumerate(registros):
            with cols[idx % 3]:
                st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                try:
                    st.image(f"data/captures/{name}.jpg", width=200)
                except:
                    st.markdown("<div class='no-photo'>Sin fotografía</div>", unsafe_allow_html=True)
                st.markdown(f"**{name}**")
                st.caption(f"Registrado: {created_at or 'Sin fecha'}")
                st.markdown("</div>", unsafe_allow_html=True)
