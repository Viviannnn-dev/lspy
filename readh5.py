import h5py
import os  # Asegúrate de importar el módulo os

file_path = r"C:\Users\LENOVO T480S\avatarpypy\modelo_lstm_lsp\data\keypoints\buenos_dias.h5"

# Verificar si el archivo existe
if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as f:
        print("Contenido del archivo:")
        
        # Acceder al grupo 'data'
        data_group = f['data']
        
        # Ver los datasets dentro del grupo 'data'
        for key in data_group.keys():
            print(f"Dataset encontrado: {key}")
            dataset = data_group[key]
            print(f"Contenido del dataset: {dataset[:]}")  # Mostrar los datos del dataset
else:
    print(f"El archivo {file_path} no se encuentra.")
