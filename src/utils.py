import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def init_camera(width=640, height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def draw_face_mesh_green(frame, results):
    if results.multi_face_landmarks:
        # Estilo personalizado para malla verde
        tesselation_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
        tesselation_style.color = (0, 255, 0)  # Verde

        contours_style = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Verde
            thickness=1,
            circle_radius=1
        )

        for face_landmarks in results.multi_face_landmarks:
            # Dibujar la teselación (malla completa)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=tesselation_style)

            # Dibujar los contornos
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=contours_style
            )
    return frame

def draw_similarity_bar(frame, confidence, x=30, y= 110, w=260, h=18):
    pct = max(0.0, min(1.0, confidence))

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(pct * w)
    cv2.rectangle(frame, (x + 2, y + 2), (x + fill_w - 2, y + h - 2), (0, 255, 0), -1)

    txt = f"{round(pct * 100, 1)}%"
    cv2.putText(frame, txt, (x + w + 10, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
                )
    return frame

def enhance_lighting(frame, apply_filter=False):
    if not apply_filter:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def auto_zoom_on_face(frame, face_landmarks, zoom_factor=1.0, maintain_size=True):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
    min_x, max_x = max(min(xs), 0), min(max(xs), w)
    min_y, max_y = max(min(ys), 0), min(max(ys), h)
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    box_w = int((max_x - min_x) * zoom_factor)
    box_h = int((max_y - min_y) * zoom_factor)
    x1, y1 = max(cx - box_w // 2, 0), max(cy - box_h // 2, 0)
    x2, y2 = min(cx + box_w // 2, w), min(cy + box_h // 2, h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    if maintain_size:
        return cv2.resize(cropped, (w, h))
    return cropped

def inject_theme():
    css = """
    <style>
    .main { background: #0f111a; }
    .stButton>button, .stCheckbox>label {
        border-radius: 8px;
    }
    .info-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: #fff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 12px 20px rgba(0,0,0,0.3);
    }
    .info-card img {
        border-radius: 12px;
        border: 2px solid rgba(255,255,255,0.5);
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .info-card .no-photo {
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 600;
    }
    </style>
    """
    return css
