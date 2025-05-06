import os
import cv2
import numpy as np
import itertools
from auxiliar_func_features import (
    detectar_caracteristicas,
    emparejar_features,
    calcular_homografia_ransac,
    calcular_homografia_ransac_manual,
    dibujar_inliers,
    crear_panorama
)

def cargar_imagenes_desde_carpeta(carpeta):
    extensiones = ['.jpg', '.jpeg', '.png']
    imagenes = [
        os.path.join(carpeta, f)
        for f in sorted(os.listdir(carpeta))
        if os.path.splitext(f)[1].lower() in extensiones
    ]
    return [cv2.imread(ruta) for ruta in imagenes]

def es_homografia_valida(H):
    return H is not None and not (np.isnan(H).any() or np.isinf(H).any())

def construir_panorama(imagenes, metodo="SIFT", nfeatures=2000, manual_homografia=False):
    # Calcular todas las combinaciones posibles de pares y puntuar
    n = len(imagenes)
    usado = [False] * n
    pares = []

    for (i, j) in itertools.combinations(range(n), 2):
        kp1, desc1, _ = detectar_caracteristicas(imagenes[i], metodo, nfeatures)
        kp2, desc2, _ = detectar_caracteristicas(imagenes[j], metodo, nfeatures)

        if desc1 is None or desc2 is None:
            continue
        
        matches, _ = emparejar_features(desc1, desc2, metodo="brute-force", tipo="NN")

        if manual_homografia:
            H, inliers = calcular_homografia_ransac_manual(kp1, kp2, matches)
        else:
            H, inliers = calcular_homografia_ransac(kp1, kp2, matches)

        if es_homografia_valida(H) and len(inliers) > 10:
            pares.append((i, j, len(inliers), H))

        # Dibujar inliers
        nombre = f"inlier_{metodo}_{i}_{j}.png"
        dibujar_inliers(imagenes[i], kp1, imagenes[j], kp2, matches, inliers, nombre)
        
    if not pares:
        print("No se encontraron combinaciones válidas.")
        return imagenes[0]

    # Ordenamos los pares por número de inliers (mayor es mejor)
    pares.sort(key=lambda x: x[2], reverse=True)

    # Unimos primero el mejor par
    i, j, _, H = pares[0]
    panorama = crear_panorama(imagenes[i], imagenes[j], H)
    usado[i] = usado[j] = True

    # Intentar unir los restantes progresivamente
    for idx in range(n):
        if usado[idx]:
            continue
        kp1, desc1, _ = detectar_caracteristicas(panorama, "SIFT", 2000)
        kp2, desc2, _ = detectar_caracteristicas(imagenes[idx], "SIFT", 2000)

        if desc1 is None or desc2 is None:
            continue

        matches, _ = emparejar_features(desc1, desc2, metodo="brute-force", tipo="NN")
        H, inliers = calcular_homografia_ransac(kp1, kp2, matches)

        if es_homografia_valida(H) and len(inliers) > 10:
            panorama = crear_panorama(panorama, imagenes[idx], H)
            usado[idx] = True

    return panorama

# ===== MAIN =====
if __name__ == "__main__":
    carpeta = "BuildingScene3"  

    metodo = "SIFT"  # "SIFT", "ORB", "AKAZE"
    nfeatures = 2350

    imagenes = cargar_imagenes_desde_carpeta(carpeta)
    if len(imagenes) < 2:
        print("Se necesitan al menos dos imágenes.")
        exit()

    panorama_final = construir_panorama(imagenes, metodo, nfeatures)

    cv2.imwrite("results/panorama_res_" + metodo + "_" + str(nfeatures) + ".png", panorama_final)
    print("Panorama guardado en results/panorama_res_" + metodo + "_" + str(nfeatures) + ".png")
