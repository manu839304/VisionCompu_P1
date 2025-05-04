import cv2
import numpy as np
import time
import os

def cargar_imagenes(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("No se pudieron cargar las imágenes. Verifica las rutas.")
    return img1, img2

def inicializar_detector(nombre, nfeatures=500):
    if nombre == 'ORB':
        return cv2.ORB_create(nfeatures=nfeatures)
    elif nombre == 'SIFT':
        return cv2.SIFT_create(nfeatures=nfeatures)
    elif nombre == 'AKAZE':
        return cv2.AKAZE_create()
    else:
        raise ValueError("Detector no reconocido")

def detectar_caracteristicas(imagen, nombre_detector, nfeatures=500):
    if nombre_detector == 'HARRIS':
        return detectar_harris(imagen=imagen, nfeatures=nfeatures)
    else:
        detector = inicializar_detector(nombre=nombre_detector, nfeatures=nfeatures)
        inicio = time.time()
        keypoints, descriptors = detector.detectAndCompute(imagen, None)
        tiempo = time.time() - inicio
        return keypoints, descriptors, tiempo

def detectar_harris(imagen, nfeatures):
    inicio = time.time()
    
    puntos = cv2.goodFeaturesToTrack(
        imagen,
        maxCorners=nfeatures,
        useHarrisDetector=True,
    )

    if puntos is None:
        return [], None, time.time() - inicio

    keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 3) for p in puntos]

    # HARRIS no genera descriptores, usamos ORB para ello
    orb = inicializar_detector('ORB', nfeatures=nfeatures)
    keypoints, descriptors = orb.compute(imagen, keypoints)

    tiempo = time.time() - inicio
    return keypoints, descriptors, tiempo


def emparejar_features(desc1, desc2, metodo='brute-force', ratio=0.75, tipo='NN'):
    # Detecta si los descriptores son binarios (ORB, AKAZE) o flotantes (SIFT)
    if desc1.dtype == np.uint8:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    if metodo == 'brute-force':
        if tipo == 'NN':
            bf = cv2.BFMatcher(norm_type, crossCheck=True)
            inicio = time.time()
            matches = bf.match(desc1, desc2)
            tiempo = time.time() - inicio
            return sorted(matches, key=lambda x: x.distance), tiempo
        elif tipo == 'NNDR':
            bf = cv2.BFMatcher(norm_type)
            inicio = time.time()
            matches = bf.knnMatch(desc1, desc2, k=2)
            tiempo = time.time() - inicio
            buenos = [m for m, n in matches if m.distance < ratio * n.distance]
            return buenos, tiempo


def mostrar_emparejamientos(img1, kp1, img2, kp2, matches, nombre="Emparejamiento", mostrar=False):
    if not os.path.exists("results"):
        os.makedirs("results")
    resultado = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if mostrar:
        cv2.imshow(nombre, resultado)
        cv2.imwrite(f"results/imagenes_emparejadas/{nombre}.jpg", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(f"results/imagenes_emparejadas/{nombre}.jpg", resultado)

def imprimir_estadisticas(tiempo_det1, tiempo_det2, tiempo_match, kp1, kp2, matches, nombre_archivo="resultados.txt", mostrar=False):
    stats = (
        f"Tiempo detección imagen 1: {tiempo_det1:.4f} s\n"
        f"Tiempo detección imagen 2: {tiempo_det2:.4f} s\n"
        f"Tiempo emparejamiento: {tiempo_match:.4f} s\n"
        f"Puntos detectados en img1: {len(kp1)}\n"
        f"Puntos detectados en img2: {len(kp2)}\n"
        f"Número de emparejamientos: {len(matches)}\n"
    )
    if mostrar:
        print(stats)
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write(stats)


if __name__ == "__main__":

    # ==== CONFIGURACIÓN ====

    #ruta_img1 = 'BuildingScene/building1.jpg'
    #ruta_img2 = 'BuildingScene/building2.jpg'
    ruta_img1 = 'BuildingScene2/img2.jpg'
    ruta_img2 = 'BuildingScene2/img3.jpg'
    
    detector_nombre = 'SIFT'      # 'ORB', 'SIFT', 'AKAZE', 'HARRIS'
    tipo_emparejamiento = 'NN'      # 'NN' para fuerza bruta directa o 'NNDR' para ratio
    nfeatures = 500                 # Solo afecta a ORB y SIFT

    # ==== PROCESAMIENTO ====

    img1, img2 = cargar_imagenes(ruta_img1, ruta_img2)
    
    kp1, desc1, tiempo1 = detectar_caracteristicas(img1, detector_nombre, nfeatures)
    kp2, desc2, tiempo2 = detectar_caracteristicas(img2, detector_nombre, nfeatures)

    matches, tiempo_match = emparejar_features(desc1, desc2, metodo='brute-force', tipo=tipo_emparejamiento)

    nombre_emparejamiento = f"{detector_nombre}_{tipo_emparejamiento}"
    mostrar_emparejamientos(img1, kp1, img2, kp2, matches, f"Emparejamiento_{nombre_emparejamiento}")

    imprimir_estadisticas(tiempo1, tiempo2, tiempo_match, kp1, kp2, matches, f"results/estadisticas_{nombre_emparejamiento}.txt")
