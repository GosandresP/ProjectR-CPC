import sys
from pathlib import Path

# Añadir la raíz del proyecto al path para permitir imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import warnings; warnings.filterwarnings("ignore")
import cv2, mediapipe as mp
import numpy as np
from utils import  init_camera, draw_face_mesh_green, draw_similarity_bar
from db.database import init_db, insert_face
from recognition import compare_faces


mp_face_mesh = mp.solutions.face_mesh

# Parametros de la camara
THRESHOLD = 0.6  # umbral de similitud para reconocer un rostro
ALPHA_SMOOTH = 0.5 # Suavizado de la barra de confianza

# Definir la funciona main principal
def main():
    init_db()
    cap = init_camera()

    # Estado para suavizado
    smooth_confidence = 0.0
    last_name = "Desconocido"

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Procesamiento continuo -> para video
            max_num_faces=3,  # Numero maximo de rostros a detectar
            refine_landmarks=True,  # Mayor detalle en ojos y labios
            # Nivel de confianza
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ok, frame = cap.read()  # leer un frame de la camara
            if not ok:
                print("No se pudo acceder a la camara")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            frame = draw_face_mesh_green(frame, results)
            curr_conf = 0.0
            curr_name = "Desconocido"

            # Extracción de los landmarks faciales
            if results.multi_face_landmarks:
                # Solo usamos el primer rostro detectado a la barra global

                first = results.multi_face_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]

                curr_name, dist, _ = compare_faces(landmarks, threshold=THRESHOLD)
                curr_conf = max(0.0, min(1.0, 1.0 - dist))  # Convertir distancia a confianza

                # Suavizado de la confianza
                smooth_confidence = (ALPHA_SMOOTH * curr_conf) + (1 - ALPHA_SMOOTH) * smooth_confidence

                if curr_name != "Desconocido":
                    last_name = curr_name

                # Overlay de TeXTOS
                cv2.putText(frame, f"{curr_name}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confianza:", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Barra de Similitud -> verde
                frame = draw_similarity_bar(frame, smooth_confidence, x=30, y=110, w=260, h=18)

            # Ventana
            cv2.imshow("Reconocimento facial (R = Registrar, Q = Salir)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if results.multi_face_landmarks:
                    first = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]
                    name = input("Nombre del usuario: ")
                    if name:
                        insert_face(name, landmarks)
                        cv2.imwrite(f"data/captures/{name}.png", frame)
                        print(f"Rostro registrado con nombre: {name}")
                        last_name = name
                        smooth_confidence = 1.0
                    else:
                        print("Registro Cancelado: Nombre Vacio")
                else:
                    print("No hay rostro visible para registrar")

    # Liberar la camara y cerrar ventanas al finalizar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
