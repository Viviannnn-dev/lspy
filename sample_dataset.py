import os
import pandas as pd
import cv2
from mediapipe.python.solutions.holistic import Holistic
from constants import *
from helpers import *

# CREACIÓN DE KEYPOINTS PARA CADA PALABRA
def create_keypoints(word_id, words_path, hdf_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `hdf_path`
    '''
    data = pd.DataFrame([])  # Crear un DataFrame vacío para almacenar los keypoints
    frames_path = os.path.join(words_path, word_id)  # Ruta de la carpeta con las muestras de la palabra
    
    with Holistic() as holistic:  # Inicializar el modelo Holistic de MediaPipe
        print(f'Creando keypoints de "{word_id}"...')
        sample_list = os.listdir(frames_path)  # Listar los archivos en la carpeta de la palabra
        sample_count = len(sample_list)
        
        # Recorrer cada muestra (ejemplo) dentro de la carpeta de la palabra
        for n_sample, sample_name in enumerate(sample_list, start=1):
            sample_path = os.path.join(frames_path, sample_name)  # Ruta de cada ejemplo
            keypoints_sequence = get_keypoints(holistic, sample_path)  # Extraer los keypoints
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)  # Insertar los keypoints en el DataFrame
            print(f"{n_sample}/{sample_count}", end="\r")  # Progreso de la creación de keypoints

    # Guardar los keypoints en el archivo HDF5
    data.to_hdf(hdf_path, key="data", mode="w")
    print(f"Keypoints creados! ({sample_count} muestras)", end="\n")

# Función para procesar todas las palabras
def process_data_from_folders(root_path, excel_path):
    '''
    Función que recorre las carpetas con ejemplos de las palabras y procesa los keypoints
    a partir de los datos del archivo Excel.
    '''
    # Leer archivo Excel con las palabras
    excel_data = pd.read_excel(excel_path)
    word_ids = [word for word in os.listdir(root_path)]
    
    # Recorrer las carpetas de las palabras
    for word_id in word_ids:
        # Obtener nombre de la palabra desde el excel
        word_name = excel_data.iloc[int(word_id)-1, 1]  # Asumimos que el ID de la palabra es un número entero
        print(f"Procesando la palabra: {word_name}")

        # Ruta donde se guardan los keypoints
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_name}.h5")
        
        # Crear keypoints para la palabra
        create_keypoints(word_id, root_path, hdf_path)

# Llamada principal para procesar los datos
if __name__ == "__main__":
    # Directorios
    root_path = "C:\\Users\\LENOVO T480S\\lspy\\MSLwords1"  # Ruta principal de la carpeta
    excel_path = 'C:\\Users\\LENOVO T480S\\lspy\\MSLwords1\\classes.xlsx'  # Ruta de tu archivo Excel

    # Crear keypoints para las palabras
    process_data_from_folders(root_path, excel_path)
    
# Función para obtener keypoints (modificada para verificar si se puede leer la imagen)
def get_keypoints(model, sample_path):
    '''
    ### OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra
    '''
    kp_seq = np.array([])
    for img_name in os.listdir(sample_path):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Error al leer la imagen: {img_path}")
            continue  # Salta este ciclo si la imagen no se lee correctamente
        
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq
