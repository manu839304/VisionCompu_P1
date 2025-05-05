import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from auxiliar_func_features import (
    cargar_imagenes,
    detectar_caracteristicas,
    emparejar_features,
    mostrar_emparejamientos,
    imprimir_estadisticas,
    calcular_homografia_ransac,
    dibujar_inliers,
    crear_panorama
)

# Métodos y parámetros a evaluar
metodos = {
    "ORB": {"params": {"nfeatures": [300, 500, 1000, 2000, 3000, 10000]}},
    "AKAZE": {"params": {"dummy": [None]}},  # AKAZE no tiene parámetro nfeatures
    "SIFT": {"params": {"nfeatures": [300, 500, 1000, 2000, 3000, 10000]}},
    "HARRIS": {"params": {"nfeatures": [300, 500, 1000, 2000, 3000, 10000]}},
}

tipos_emparejamiento = ["NN", "NNDR"]

# Cargar imágenes
img1, img2 = cargar_imagenes('BuildingScene/building1.jpg', 'BuildingScene/building2.jpg')

# Crear carpetas si no existen
os.makedirs("results/estadisticas", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)
os.makedirs("results/graficas", exist_ok=True)
os.makedirs("results/imagenes_inliers", exist_ok=True)
os.makedirs("results/panoramas", exist_ok=True)

# Resultados acumulados
resultados = []

# Comparaciones
for metodo, config in metodos.items():
    for param, valores in config['params'].items():
        for valor in valores:
            for tipo in tipos_emparejamiento:
                if metodo == 'AKAZE':
                    nombre = f"{metodo}_{tipo}"
                else:
                    nombre = f"{metodo}_{valor}_{tipo}"

                # Calculamos las características
                kp1, desc1, tiempo1 = detectar_caracteristicas(img1, metodo, valor)
                kp2, desc2, tiempo2 = detectar_caracteristicas(img2, metodo, valor)

                if desc1 is None or desc2 is None:
                    print(f"{nombre}: Descriptores no encontrados.")
                    continue

                # Emparejamos las características
                try:
                    matches, tiempo_match = emparejar_features(desc1, desc2, metodo='brute-force', tipo=tipo)
                except Exception as e:
                    print(f"Error emparejando {nombre}: {e}")
                    continue

                mostrar_emparejamientos(img1, kp1, img2, kp2, matches, f"{nombre}")

                # Guardamos resultados
                imprimir_estadisticas(
                    tiempo1, tiempo2, tiempo_match, kp1, kp2, matches,
                    f"results/estadisticas/estadisticas_{nombre}.txt"
                )

                resultados.append({
                    "Metodo": metodo,
                    "Parametro": valor if metodo != 'AKAZE' else '-',
                    "Tipo": tipo,
                    "Tiempo1": tiempo1,
                    "Tiempo2": tiempo2,
                    "Tiempo_Match": tiempo_match,
                    "Keypoints1": len(kp1),
                    "Keypoints2": len(kp2),
                    "Matches": len(matches)
                })

                # --------- PANORAMA ---------
                # Calculamos homografia
                H, inliers = calcular_homografia_ransac(kp1, kp2, matches)
                if H is None:
                    print("No se pudo calcular homografía")
                    continue
                
                dibujar_inliers(img1, kp1, img2, kp2, matches, inliers, nombre="inliers_" + nombre)

                panorama = crear_panorama(img1, img2, H)
                cv2.imwrite(f"results/panoramas/panorama_{nombre}.jpg", panorama)


# Convertir a DataFrame
df = pd.DataFrame(resultados)
df.to_csv("results/tablas/resumen_resultados.csv", index=False)
print("Resumen guardado en results/tablas/resumen_resultados.csv")
