import os
import cv2
import json
import time
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

UMBRAL_CONFIANZA = 0.60   # Por debajo de esto la predicción se marca como incierta
AREA_MINIMA = 0.03        # Fracción mínima del recuadro que debe ocupar la prenda
MODOS_FONDO = ["AUTO", "FONDO CLARO", "FONDO OSCURO"]


def preprocesar(roi, modo_fondo="AUTO"):
    """Convierte el recuadro de la cámara al formato de Fashion MNIST.

    Fashion MNIST tiene la prenda clara, centrada y sobre fondo negro, ocupando
    todo el alto o ancho de la imagen de 28x28. Pasos:
      1. Grises + suavizado y umbral de Otsu para separar prenda y fondo.
      2. El fondo es lo que toca los bordes del recuadro (modo AUTO).
      3. Se queda el contorno más grande, se recorta y se pone el fondo en negro.
      4. Se escala el lado mayor a 28 px y se centra en un lienzo negro de 28x28.

    Devuelve (imagen_28x28 uint8, contorno en coordenadas del roi) o (None, None)
    si no se encontró una prenda.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    suave = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mascara = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # La prenda debe quedar en blanco dentro de la máscara
    if modo_fondo == "AUTO":
        borde = np.concatenate([mascara[0], mascara[-1], mascara[:, 0], mascara[:, -1]])
        fondo_claro = borde.mean() > 127
    else:
        fondo_claro = modo_fondo == "FONDO CLARO"
    if fondo_claro:
        mascara = cv2.bitwise_not(mascara)
        prenda = cv2.bitwise_not(gray)  # Prenda oscura sobre fondo claro -> invertir tonos
    else:
        prenda = gray

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None
    contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno) < AREA_MINIMA * roi.shape[0] * roi.shape[1]:
        return None, None

    # Rellenar el contorno para no perder el interior de la prenda (estampados, sombras)
    mascara_prenda = np.zeros_like(mascara)
    cv2.drawContours(mascara_prenda, [contorno], -1, 255, cv2.FILLED)
    prenda = cv2.bitwise_and(prenda, prenda, mask=mascara_prenda)

    x, y, w, h = cv2.boundingRect(contorno)
    recorte = prenda[y:y + h, x:x + w]

    # Estirar el contraste para que la prenda sea clara como en el dataset
    valores = recorte[mascara_prenda[y:y + h, x:x + w] > 0]
    bajo, alto = np.percentile(valores, [2, 98])
    recorte = np.clip((recorte.astype("float32") - bajo) / max(alto - bajo, 1) * 205 + 50, 0, 255)
    recorte[mascara_prenda[y:y + h, x:x + w] == 0] = 0
    recorte = recorte.astype("uint8")

    # Lado mayor a 28 px manteniendo la proporción y centrar en lienzo negro
    escala = 28 / max(w, h)
    nw, nh = max(1, round(w * escala)), max(1, round(h * escala))
    pequeno = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_AREA)
    lienzo = np.zeros((28, 28), dtype="uint8")
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    lienzo[oy:oy + nh, ox:ox + nw] = pequeno
    return lienzo, contorno


def predecir(model, img28):
    tensor_input = img28[np.newaxis, ..., np.newaxis].astype("float32")
    return model(tensor_input, training=False).numpy()[0]


def dibujar_texto(frame, texto, pos, escala, color, grosor=2):
    # Contorno negro para que el texto se lea sobre cualquier fondo
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def abrir_camara(indice):
    """Abre la cámara indicada; devuelve None si no existe o no entrega imagen."""
    cap = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Descartar los primeros cuadros mientras la cámara ajusta la exposición (salen negros)
    for _ in range(30):
        ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap


def main():
    print(f"Cargando modelo local: {mejor_modelo_nombre}...")
    model = keras.models.load_model("modelos/mejor_modelo.keras")
    print("¡Modelo cargado exitosamente!")

    # 2. Iniciar captura de cámara web nativa (0 es la cámara integrada por defecto).
    # Con la tecla 'c' se pasa a la siguiente cámara (por ejemplo, el iPhone con Continuity Camera)
    CAMARA_INDEX = 0
    MAX_CAMARAS = 4
    cap = abrir_camara(CAMARA_INDEX)
    if cap is None:
        print("Error: No se pudo acceder a la cámara web.")
        exit()

    # Crear la ventana por adelantado y mantenerla al frente (en macOS se abre detrás del editor)
    NOMBRE_VENTANA = "Evaluacion de Prendas en Tiempo Real"
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_TOPMOST, 1)

    modo_fondo = 0
    tam_recuadro = 0.8  # Fracción del alto del cuadro que ocupa el recuadro (teclas + y -)
    congelado = False
    os.makedirs("capturas", exist_ok=True)

    print("--- INSTRUCCIONES EN VIVO ---")
    print("Coloque la prenda dentro del recuadro, sobre un fondo liso y con buena luz")
    print("Presiona 'c' para cambiar de cámara (Mac / iPhone)")
    print("Presiona '+' o '-' para agrandar o achicar el recuadro")
    print("Presiona 'i' para cambiar el tipo de fondo (Auto / Claro / Oscuro)")
    print("Presiona ESPACIO para congelar la predicción")
    print("Presiona 's' para guardar una captura")
    print("Presiona 'q' para salir de la aplicación")

    vista = None
    while True:
        if not congelado:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Efecto espejo, más natural para mostrar prendas

            # Región de interés central (ROI): tam_recuadro del alto del cuadro
            h, w, _ = frame.shape
            crop_sz = int(min(h, w) * tam_recuadro)
            start_x, start_y = (w - crop_sz) // 2, (h - crop_sz) // 2
            roi = frame[start_y:start_y + crop_sz, start_x:start_x + crop_sz].copy()

            img28, contorno = preprocesar(roi, MODOS_FONDO[modo_fondo])

            if img28 is None:
                b_color = (0, 0, 255)
                lineas = [("Sin prenda en el recuadro", b_color)]
            else:
                preds = predecir(model, img28)
                top3 = np.argsort(preds)[::-1][:3]
                confianza = preds[top3[0]]
                b_color = (0, 255, 0) if confianza >= UMBRAL_CONFIANZA else (0, 165, 255)
                principal = f"{CLASES_ES[top3[0]]} ({confianza:.1%})"
                if confianza < UMBRAL_CONFIANZA:
                    principal = f"Incierto: {principal}"
                lineas = [(principal, b_color)] + [
                    (f"{CLASES_ES[c]} ({preds[c]:.1%})", (220, 220, 220)) for c in top3[1:]]
                contorno = contorno + np.array([start_x, start_y])
                cv2.drawContours(frame, [contorno], -1, (255, 255, 0), 2)

            # Feedback visual (verde: confianza alta, naranja: incierta, rojo: sin prenda)
            cv2.rectangle(frame, (start_x, start_y), (start_x + crop_sz, start_y + crop_sz), b_color, 3)
            dibujar_texto(frame, "Coloque la prenda aqui", (start_x + 10, start_y + 30), 0.7, (255, 255, 255))
            # Predicción y top-3 fijos arriba a la izquierda, para que se vean con cualquier tamaño de recuadro
            dibujar_texto(frame, lineas[0][0], (10, 75), 1.0, lineas[0][1], 2)
            for i, (texto, color) in enumerate(lineas[1:]):
                dibujar_texto(frame, texto, (10, 105 + 25 * i), 0.6, color, 1)
            dibujar_texto(frame, f"Camara: {CAMARA_INDEX} (c)   Fondo: {MODOS_FONDO[modo_fondo]} (i)",
                          (10, 30), 0.7, (255, 200, 0))

            # Miniatura exacta que recibe la red, en la esquina superior derecha
            if img28 is not None:
                mini = cv2.resize(img28, (168, 168), interpolation=cv2.INTER_NEAREST)
                frame[45:45 + 168, w - 178:w - 10] = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame, (w - 178, 45), (w - 10, 45 + 168), (255, 255, 255), 1)
                dibujar_texto(frame, "Entrada CNN 28x28", (w - 178, 35), 0.5, (255, 255, 255), 1)
            vista = frame

        mostrar = vista.copy()
        if congelado:
            dibujar_texto(mostrar, "CONGELADO (ESPACIO para continuar)", (10, mostrar.shape[0] - 20), 0.7, (0, 255, 255))
        cv2.imshow(NOMBRE_VENTANA, mostrar)

        # Manejo de teclado nativo
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            modo_fondo = (modo_fondo + 1) % len(MODOS_FONDO)
        elif key in (ord('+'), ord('=')):
            tam_recuadro = min(1.0, tam_recuadro + 0.1)
        elif key in (ord('-'), ord('_')):
            tam_recuadro = max(0.4, tam_recuadro - 0.1)
        elif key == ord('c'):
            # Probar las siguientes cámaras hasta encontrar una que funcione
            cap.release()
            for paso in range(1, MAX_CAMARAS + 1):
                indice = (CAMARA_INDEX + paso) % MAX_CAMARAS
                nueva = abrir_camara(indice)
                if nueva is not None:
                    cap, CAMARA_INDEX = nueva, indice
                    print("Usando cámara", CAMARA_INDEX)
                    break
            congelado = False
        elif key == ord(' '):
            congelado = not congelado
        elif key == ord('s'):
            ruta = os.path.join("capturas", time.strftime("captura_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(ruta, mostrar)
            print("Captura guardada en", ruta)

    # Liberar recursos de cámara al finalizar
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada.")


if __name__ == "__main__":
    main()
