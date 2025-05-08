import os
import cv2
import numpy as np
import itertools
import time
import argparse
import warnings
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

def guardar_metricas_panorama(metodo, nfeatures, imagenes, pares_validos, imagenes_usadas, tiempos, tam_panorama, nombre_archivo):

    carpeta_metricas = "results/metricas_panoramas"
    os.makedirs(carpeta_metricas, exist_ok=True)
    
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write(f"Método de detección: {metodo}\n")
        f.write(f"Número de características: {nfeatures}\n")
        f.write(f"Número total de imágenes: {len(imagenes)}\n")
        f.write(f"Número de pares válidos con homografía: {len(pares_validos)}\n\n")

        f.write("Pares válidos (i, j, número de inliers):\n")
        for i, j, inliers, _ in pares_validos:
            f.write(f"  ({i}, {j}) - Inliers: {inliers}\n")
        
        f.write("\nOrden y total de imágenes usadas en el panorama:\n")
        usadas = [i for i, val in enumerate(imagenes_usadas) if val]
        f.write(f"  Índices: {usadas}\n")
        f.write(f"  Total: {len(usadas)}\n")

        f.write("\nTiempos por fase:\n")
        for clave, valor in tiempos.items():
            f.write(f"  {clave}: {valor:.4f} segundos\n")

        total_inliers = sum(p[2] for p in pares_validos)
        media_inliers = total_inliers / len(pares_validos) if pares_validos else 0
        num_pares_posibles = len(imagenes) * (len(imagenes) - 1) / 2
        tasa_validos = len(pares_validos) / num_pares_posibles if num_pares_posibles > 0 else 0

        f.write("\nMétricas adicionales:\n")
        f.write(f"  Inliers totales: {total_inliers}\n")
        f.write(f"  Inliers promedio por par: {media_inliers:.2f}\n")
        f.write(f"  Tasa de pares válidos: {tasa_validos:.2f}\n")
        if tam_panorama:
            f.write(f"  Tamaño del panorama: {tam_panorama[0]} x {tam_panorama[1]} px\n")

def es_homografia_valida(H):
    return H is not None and not (np.isnan(H).any() or np.isinf(H).any())

def construir_panorama(imagenes, metodo="SIFT", nfeatures=2000, manual_homografia=False):
    tiempos = {}
    t_total = time.time()

    carpeta_resultados_prog = "results/imagenes_progresivas"
    os.makedirs(carpeta_resultados_prog, exist_ok=True)

    carpeta_resultados = "results/panoramas"
    os.makedirs(carpeta_resultados, exist_ok=True)

    n = len(imagenes)
    usado = [False] * n
    pares = []

    t_det = time.time()
    keypoints_descs = [detectar_caracteristicas(img, metodo, nfeatures) for img in imagenes]
    tiempos["detección"] = time.time() - t_det

    t_match = time.time()
    for (i, j) in itertools.combinations(range(n), 2):
        kp1, desc1, _ = keypoints_descs[i]
        kp2, desc2, _ = keypoints_descs[j]

        if desc1 is None or desc2 is None:
            continue

        matches, _ = emparejar_features(desc1, desc2, metodo="brute-force", tipo="NN")

        if manual_homografia:
            H, inliers = calcular_homografia_ransac_manual(kp1, kp2, matches)
        else:
            H, inliers = calcular_homografia_ransac(kp1, kp2, matches)

        if es_homografia_valida(H) and len(inliers) > 10:
            pares.append((i, j, len(inliers), H))
            nombre = f"inlier_{metodo}_{i}_{j}"
            dibujar_inliers(imagenes[i], kp1, imagenes[j], kp2, matches, inliers, nombre)
    tiempos["emparejamiento + homografías"] = time.time() - t_match

    if not pares:
        print("No se encontraron combinaciones válidas.")
        return imagenes[0]

    pares.sort(key=lambda x: x[2], reverse=True)

    i, j, _, H = pares[0]
    panorama = crear_panorama(imagenes[i], imagenes[j], H)
    usado[i] = usado[j] = True

    cv2.imwrite(os.path.join(carpeta_resultados_prog, "panorama_paso_1.png"), panorama)
    paso = 2

    t_cosido = time.time()
    for idx in range(n):
        if usado[idx]:
            continue
        kp1, desc1, _ = detectar_caracteristicas(panorama, metodo, nfeatures)
        kp2, desc2, _ = keypoints_descs[idx]

        if desc1 is None or desc2 is None:
            continue

        matches, _ = emparejar_features(desc1, desc2, metodo="brute-force", tipo="NN")
        H, inliers = calcular_homografia_ransac(kp1, kp2, matches)

        if es_homografia_valida(H) and len(inliers) > 10:
            panorama = crear_panorama(panorama, imagenes[idx], H)
            usado[idx] = True
            cv2.imwrite(os.path.join(carpeta_resultados_prog, f"panorama_paso_{paso}.png"), panorama)
            paso += 1
    tiempos["cosido"] = time.time() - t_cosido

    tiempos["tiempo_total"] = time.time() - t_total

    tam_panorama = panorama.shape[:2] if panorama is not None else None

    nombre_archivo = f"results/metricas_panoramas/datos_panorama_res_{metodo}_{nfeatures}.txt"
    guardar_metricas_panorama(
        metodo=metodo,
        nfeatures=nfeatures,
        imagenes=imagenes,
        pares_validos=pares,
        imagenes_usadas=usado,
        tiempos=tiempos,
        tam_panorama=tam_panorama,
        nombre_archivo=nombre_archivo
    )
    print(f"Métricas guardadas en {nombre_archivo}")

    return panorama

# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("carpeta", help="Carpeta con imágenes (por ejemplo, VPG/S33)")
    parser.add_argument("--metodo", type=str, default="SIFT", choices=["SIFT", "ORB", "AKAZE"],
                        help="Método de detección de características")
    parser.add_argument("--nfeatures", type=int, default=2000,
                        help="Número de características a detectar (positivo, >0)")
    args = parser.parse_args()

    if not os.path.exists(args.carpeta):
        raise FileNotFoundError(f"La carpeta especificada no existe: {args.carpeta}")

    if args.nfeatures <= 0:
        raise ValueError("El número de características debe ser un entero positivo.")

    imagenes = cargar_imagenes_desde_carpeta(args.carpeta)

    if len(imagenes) < 2:
        warnings.warn("Se necesitan al menos dos imágenes válidas para construir un panorama.")
        exit()

    if len(imagenes) > 8:
        warnings.warn("Hay más de 8 imágenes, esto puede alargar significativamente el tiempo de procesamiento.")

    print(f"Usando método: {args.metodo}, características: {args.nfeatures}, carpeta: {args.carpeta}")
    panorama_final = construir_panorama(imagenes, metodo=args.metodo, nfeatures=args.nfeatures)

    os.makedirs("results/panoramas", exist_ok=True)
    ruta_salida = f"results/panoramas/panorama_res_{args.metodo}_{args.nfeatures}.png"
    cv2.imwrite(ruta_salida, panorama_final)
    print(f"Panorama guardado en: {ruta_salida}")