import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. Cargar metadatos de clases y modelo
try:
    with open("modelos/clases.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    CLASES_ES = metadata["es"]
    mejor_modelo_nombre = metadata["mejor_modelo"]
except Exception:
    CLASES_ES = ["Camiseta", "Pantalon", "Sueter", "Vestido", "Abrigo",
                 "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botin"]
    mejor_modelo_nombre = "Mini_ResNet"

print(f"Cargando modelo local: {mejor_modelo_nombre}...")
model = keras.models.load_model("modelos/mejor_modelo.keras")
print("¡Modelo cargado exitosamente!")

# 2. Iniciar captura de cámara web nativa (0 es la cámara integrada por defecto)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    exit()

invertir_colores = True  # True para fondos claros / prendas oscuras

print("--- INSTRUCCIONES EN VIVO ---")
print("Presiona 'i' para alternar la inversión de colores (Fondo claro/oscuro)")
print("Presiona 'q' para salir de la aplicación")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Región de interés central (ROI)
    h, w, _ = frame.shape
    crop_sz = 220
    start_x, start_y = (w - crop_sz) // 2, (h - crop_sz) // 2
    roi = frame[start_y:start_y+crop_sz, start_x:start_x+crop_sz]

    # Preprocesamiento de la imagen para la CNN
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    if invertir_colores:
        resized = cv2.bitwise_not(resized)

    # Adecuar forma del tensor (1, 28, 28, 1)
    tensor_input = resized[np.newaxis, ..., np.newaxis].astype("float32")

    # Predicción
    preds = model.predict(tensor_input, verbose=0)[0]
    clase_id = np.argmax(preds)
    confianza = preds[clase_id]

    # Feedback visual (Cuadro verde para alta confianza, naranja para baja)
    b_color = (0, 255, 0) if confianza > 0.70 else (0, 165, 255)
    cv2.rectangle(frame, (start_x, start_y), (start_x+crop_sz, start_y+crop_sz), b_color, 2)

    # Mostrar textos en el visor de OpenCV
    texto_estado = "Inversion: ACTIVA" if invertir_colores else "Inversion: INACTIVA"
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.putText(frame, "Coloque la prenda aqui", (start_x, start_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    texto_pred = f"{CLASES_ES[clase_id]} ({confianza:.1%})"
    cv2.putText(frame, texto_pred, (start_x, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2, cv2.LINE_AA)

    # Renderizar ventana local interactiva
    cv2.imshow("Evaluacion de Prendas en Tiempo Real", frame)

    # Manejo de teclado nativo
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        invertir_colores = not invertir_colores

# Liberar recursos de cámara al finalizar
cap.release()
cv2.destroyAllWindows()
print("Aplicación finalizada.")
