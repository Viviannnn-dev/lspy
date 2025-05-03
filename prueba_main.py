import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.uic import loadUi

import numpy as np
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from helpers import *  # Aquí están tus helpers, como extract_keypoints y get_word_ids
from constants import *  # Constantes como MODEL_PATH y WORDS_JSON_PATH
from text_to_speech import text_to_speech  # Para que puedas hacer texto a voz

class VideoRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('mainwindow.ui', self)
        
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: No se puede acceder a la cámara.")
            sys.exit()
        
        self.init_lsp()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Actualizar frame cada 30ms
    
    def init_lsp(self):
        # Cargar el modelo preentrenado
        self.holistic_model = Holistic()
        self.model = load_model(MODEL_PATH)  # Carga el modelo de la red neuronal

        self.kp_seq = []  # Lista para almacenar la secuencia de keypoints
        self.sentence = []  # Lista para almacenar la oración traducida
        self.word_ids = get_word_ids(WORDS_JSON_PATH)  # Obtiene los ids de las palabras desde el archivo json
        self.recording = False
        self.count_frame = 0
        self.fix_frames = 0
        self.margin_frame = 1
        self.delay_frames = 3
    
    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Error al leer el frame.")
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mediapipe_detection(frame, self.holistic_model)  # Obtener puntos clave
        
        # Si se detecta mano o estamos grabando, procesamos la secuencia de keypoints
        if there_hand(results) or self.recording:
            self.recording = False
            self.count_frame += 1
            if self.count_frame > self.margin_frame:
                self.kp_seq.append(extract_keypoints(results))  # Agregar los keypoints del frame

        else:
            if self.count_frame >= MIN_LENGTH_FRAMES + self.margin_frame:
                self.fix_frames += 1
                if self.fix_frames < self.delay_frames:
                    self.recording = True
                    return

                self.kp_seq = self.kp_seq[:-(self.margin_frame + self.delay_frames)]
                kp_normalized = normalize_keypoints(self.kp_seq, int(MODEL_FRAMES))  # Normalizar la secuencia de keypoints
                res = self.model.predict(np.expand_dims(kp_normalized, axis=0))[0]  # Hacer la predicción

                # Si la predicción es confiable (por ejemplo, mayor al 70%)
                if res[np.argmax(res)] > 0.7:
                    word_id = self.word_ids[np.argmax(res)].split('-')[0]  # Obtener el id de la palabra
                    sent = words_text.get(word_id)  # Obtener el texto de la palabra correspondiente
                    self.sentence.insert(0, sent)  # Agregar la palabra a la oración
                    text_to_speech(sent)  # Reproducir el audio con la palabra
                    
                self.recording = False
                self.fix_frames = 0
                self.count_frame = 0
                self.kp_seq = []

        self.lbl_output.setText(" - ".join(self.sentence))  # Mostrar la oración en la interfaz
        draw_keypoints(image, results)  # Dibujar los keypoints en la imagen
        
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        
        scaled_qImg = qImg.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.lbl_video.setPixmap(QPixmap.fromImage(scaled_qImg))  # Actualizar el video en la interfaz

    def closeEvent(self, event):
        self.capture.release()  # Liberar la cámara cuando se cierra la ventana
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoRecorder()
    window.show()
    sys.exit(app.exec_())
