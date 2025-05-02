import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from extract_and_compare import (
    cargar_imagenes,
    inicializar_detector,
    detectar_caracteristicas,
    emparejar_features,
    mostrar_emparejamientos,
    imprimir_estadisticas
)

# Métodos y parámetros a evaluar
metodos = {
    "ORB": {"params": {"nfeatures": [300, 500, 1000, 2000, 3000, 10000]}},
    "AKAZE": {"params": {"dummy": [None]}},  # AKAZE no tiene parámetro nfeatures
    "SIFT": {"params": {"nfeatures": [300, 500, 1000, 2000, 3000, 10000]}},
    # --- FALTA HARRIS ---
}

tipos_emparejamiento = ["NN", "NNDR"]

# Cargar imágenes
img1, img2 = cargar_imagenes('BuildingScene/building1.jpg', 'BuildingScene/building2.jpg')

# Crear carpetas si no existen
os.makedirs("results/estadisticas", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)
os.makedirs("results/graficas", exist_ok=True)

# Resultados acumulados
resultados = []

# Comparaciones
for metodo, config in metodos.items():
    for param, valores in config['params'].items():
        for valor in valores:
            for tipo in tipos_emparejamiento:
                if metodo == 'AKAZE':
                    detector = inicializar_detector(metodo)
                    nombre = f"{metodo}_{tipo}"
                else:
                    detector = inicializar_detector(metodo, nfeatures=valor)
                    nombre = f"{metodo}_{valor}_{tipo}"

                # Detectar características
                kp1, desc1, tiempo1 = detectar_caracteristicas(detector, img1)
                kp2, desc2, tiempo2 = detectar_caracteristicas(detector, img2)

                if desc1 is None or desc2 is None:
                    print(f"{nombre}: Descriptores no encontrados.")
                    continue

                # Emparejar
                try:
                    matches, tiempo_match = emparejar_features(desc1, desc2, metodo='brute-force', tipo=tipo)
                except Exception as e:
                    print(f"Error emparejando {nombre}: {e}")
                    continue

                # Mostrar y guardar emparejamientos
                mostrar_emparejamientos(img1, kp1, img2, kp2, matches, f"{nombre}")

                # Guardar estadísticas
                imprimir_estadisticas(
                    tiempo1, tiempo2, tiempo_match, kp1, kp2, matches,
                    f"results/estadisticas/estadisticas_{nombre}.txt"
                )

                # Guardar en resultados
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

# Convertir a DataFrame
df = pd.DataFrame(resultados)
df.to_csv("results/tablas/resumen_resultados.csv", index=False)
print("Resumen guardado en results/tablas/resumen_resultados.csv")

# Grafica de resultados
colores_tipo = {
    "AKAZE_NN": "#1f77b4",      # Azul
    "AKAZE_NNDR": "#5fa2dd",    # Azul claro

    "ORB_NN": "#2ca02c",        # Verde
    "ORB_NNDR": "#81c784",      # Verde claro

    "SIFT_NN": "#d62728",       # Rojo
    "SIFT_NNDR": "#f1948a",     # Rojo claro
}

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

for metodo in df['Metodo'].unique():
    for tipo in tipos_emparejamiento:
        subset = df[(df['Metodo'] == metodo) & (df['Tipo'] == tipo)]
        if subset.empty:
            continue

        etiquetas_x = [
            f"{metodo}_{row['Parametro']}_{tipo}" for _, row in subset.iterrows()
        ]

        clave_color = f"{metodo}_{tipo}"
        color = colores_tipo.get(clave_color, 'gray')

        plt.plot(
            etiquetas_x,
            subset['Matches'],
            marker='o',
            linestyle='-',
            label=f"{metodo} - {tipo}",
            color=color
        )

plt.title("Comparativa del número de matches por método y tipo de emparejamiento", fontsize=14)
plt.xlabel("Configuración (Método_Parámetro_Tipo)", fontsize=12)
plt.ylabel("Número de Matches", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Método + Tipo de emparejamiento", loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("results/graficas/matches_por_metodo_tipo.png")
plt.show()