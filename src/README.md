# ProjectR - CPC

¡Bienvenido a ProjectR - CPC! 🎯

Un sistema de reconocimiento facial en tiempo real escrito en Python que usa MediaPipe para detectar y dibujar la malla facial, OpenCV para captura y visualización, y una base de datos SQLite para almacenar rostros registrados.

---

## 🔎 Resumen

- Detección y visualización de malla facial en tiempo real (MediaPipe Face Mesh).
- Registro de nuevos rostros (almacenamiento de landmarks en SQLite).
- Comparación simple por distancia entre landmarks para reconocimiento.
- Indicador visual de confianza con suavizado.

---

## 📁 Estructura del proyecto

```
ProjectR - CPC/
│
├── src/
│   ├── main.py              # Punto de entrada principal (bucle de captura y UI)
│   ├── recognition.py       # Lógica de comparación entre rostros
│   ├── utils.py             # Utilidades: cámara, dibujo de barra de confianza, etc.
│   ├── requirements.txt     # Dependencias del proyecto
│   └── README.md            # Documentación (este archivo)
│
├── db/
│   └── database.py          # Creación y consultas a la DB SQLite
│
├── data/
│   ├── face_data.db         # Base de datos SQLite con rostros
│   ├── Capturas/            # Imágenes capturadas al registrar (por nombre)
│   └── Embeddings/          # (Espacio para almacenar embeddings si se añaden)
│
└── .venv/                   # Entorno virtual (local)
```

---

## 🛠️ Instalación rápida

Abre un terminal (Windows - cmd) en la raíz del proyecto y sigue estos pasos:

1. Crear y activar entorno virtual (opcional pero recomendado):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:

```cmd
pip install -r src\requirements.txt
```

> Nota: `requirements.txt` incluye las librerías principales: `opencv-python`, `mediapipe`, `numpy`.

---

## ▶️ Ejecución

Desde la raíz del proyecto (o desde la carpeta `src`), ejecuta:

```cmd
python src\main.py
```

Controles dentro de la ventana de OpenCV:
- Presiona `R` para registrar el rostro visible (se pedirá nombre por consola).
- Presiona `Q` para salir del programa.

---

## 🧩 Descripción de archivos importantes

- `src/main.py`
  - Inicializa la base de datos y la cámara.
  - Ejecuta el bucle principal de captura, proceso con MediaPipe y renderizado con OpenCV.
  - Usa `draw_face_mesh_green(frame, results)` para dibujar la malla facial.
  - Llama a `compare_faces(...)` para identificar rostros registrados.

- `src/recognition.py`
  - Contiene la lógica para comparar los `landmarks` del rostro detectado con los almacenados en la base de datos.
  - Devuelve `(nombre, distancia)`; si no hay match, devuelve `("Desconocido", 1.0)`.

- `src/utils.py`
  - `init_camera()` — inicializa y configura la cámara (resolución, índices).
  - `draw_similarity_bar(frame, pct, x, y, w, h)` — dibuja la barra de confianza y el porcentaje.

- `db/database.py`
  - `init_db()` — crea la tabla `faces` si no existe.
  - `insert_face(name, landmarks)` — guarda un nuevo rostro (landmarks serializados).
  - `get_all_faces()` — recupera todos los rostros para comparación.

---

## ✅ Cambios y correcciones recientes

Se realizaron correcciones mínimas y necesarias sin cambiar la arquitectura principal del proyecto:

1. Eliminada la importación conflictiva `import cap` en `src/main.py` para que la variable `cap` represente correctamente el objeto de la cámara retornado por `init_camera()`.

2. Reubicado/organizado el bloque `with mp_face_mesh.FaceMesh(...) as face_mesh:` dentro de la función `main()` para que `cap` y demás variables de estado existan en el scope correcto.

3. Implementada y corregida la función `draw_face_mesh_green(frame, results)` para garantizar que:
   - Dibuja `FACEMESH_TESSELATION` y `FACEMESH_CONTOURS`.
   - El `return frame` está fuera del bucle `for`, de modo que devuelve el frame completo después de dibujar todas las caras.

4. Corregido bug en `src/utils.py` (línea que formateaba el porcentaje):
   - Antes: `txt = f"{int(pct * 100, 1)}%"` → provocaba `ValueError`.
   - Ahora: `txt = f"{round(pct * 100, 1)}%"` → muestra un decimal y evita errores.

Estos cambios fueron pensados para ser los mínimos necesarios y así mantener la lógica principal intacta.

---

## ✅ Requisitos de funcionamiento

- Python 3.8+ (probado en 3.10/3.11)
- Cámara web conectada y no utilizada por otra app
- Paquetes: ver `src/requirements.txt`

---

## 🧪 Pruebas y validación rápida

1. Activa el entorno y ejecuta `python src\main.py`.
2. Verifica que la ventana muestre la imagen de la cámara y la malla facial (si hay rostros visibles).
3. Presiona `R`, confirma nombre en consola y revisa que se guarde una imagen en `data\Capturas` y que la DB haya guardado el registro.
4. Prueba reconocimiento mostrando el mismo rostro: debería aparecer el nombre y la barra de confianza.

---

## 🐛 Solución de problemas comunes

- Error: `module 'cap' has no attribute 'read'`
  - Causa: tener `import cap` en `main.py` o conflicto con nombres. Solución: eliminar la importación y usar `cap = init_camera()`.

- Error: `ValueError: int() base must be >= 2 and <= 36, or 0`
  - Causa: uso incorrecto de `int()` con dos argumentos. Solución: usar `round(pct * 100, 1)` en `draw_similarity_bar`.

- Error: `No se pudo acceder a la camara`
  - Revisar si otra aplicación está usando la cámara o si el índice de cámara (0) es correcto.

---

## 🔭 Sugerencias de mejora (futuro)

- Reemplazar comparación de landmarks por embeddings (FaceNet / ArcFace) para mayor robustez.
- Guardar múltiples capturas por usuario con distintos ángulos.
- Añadir interfaz gráfica para gestionar la base de datos y registros.
- Soporte para múltiples cámaras y configuración por archivo `config.yaml`.

---

## 📜 Licencia

Proyecto para fines educativos. Si deseas usarlo en producción revisa y ajusta licencias de dependencias.

---

## 🤝 Contribuciones

Si quieres colaborar, abre un issue o un pull request. Aporta pruebas y descripciones claras de cambios.

---

¡Gracias por usar ProjectR - CPC! Si quieres, puedo también:
- Incluir ejemplos de `requirements.txt` o crear un script de arranque.
- Añadir un archivo `CONTRIBUTING.md` o `CHANGELOG.md`.


---

*Generado y documentado el proyecto en español — si quieres, lo adaptamos al inglés o añadimos más secciones.*

