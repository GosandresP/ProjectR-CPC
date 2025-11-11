import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def init_camera(width=640, height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def draw_face_mesh(frame, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
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