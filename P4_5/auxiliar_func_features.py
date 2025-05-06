import cv2
import numpy as np
import time
import os
import random

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
    "Detectores posibles: ORB, SIFT, AKAZE, HARRIS"
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
        minDistance=5,
        qualityLevel=0.01,
        maxCorners=nfeatures,
        useHarrisDetector=True,
    )

    if puntos is None:
        return [], None, time.time() - inicio

    keypoints = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 3) for p in puntos]

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


def calcular_homografia_ransac(kp1, kp2, matches):
    if len(matches) < 4:
        print("ERROR: Para calcular la homografia se necesitan al menos 4 matches")
        return None, [], None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    inliers = mask.ravel().tolist()
    
    return H, inliers



def calcular_homografia_ransac_manual(kp1, kp2, matches, num_iter=10000, umbral=5.0):
    if len(matches) < 4:
        print("ERROR: Para calcular la homografía se necesitan al menos 4 matches")
        return None, [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    max_inliers = []
    mejor_H = None

    for _ in range(num_iter):
        # Elegimos 4 matches aleatorios
        idxs = random.sample(range(len(matches)), 4)
        sample_pts1 = pts1[idxs]
        sample_pts2 = pts2[idxs]

        try:
            H = calcular_homografia_dlt(sample_pts2, sample_pts1)
        except np.linalg.LinAlgError: # Si surge algún error, ignoramos la iteración
            continue

        # Proyectamos todos los pts2 al sistema de pts1 usando H
        pts2_h = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)
        pts2_proj = (H @ pts2_h.T).T
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2][:, np.newaxis]

        # Calculamos distancias euclidianas entre puntos proyectados y reales 
        # y filtramos las que no superan el umbral
        dists = np.linalg.norm(pts1 - pts2_proj, axis=1)
        inliers = dists < umbral

        # Elegimos el mejor H
        if np.sum(inliers) > np.sum(max_inliers):
            max_inliers = inliers
            mejor_H = H

    if mejor_H is None or np.count_nonzero(max_inliers) < 4:
        return None, max_inliers.tolist()

    # Recalculamos H con todos los inliers
    inlier_pts1 = pts1[max_inliers]
    inlier_pts2 = pts2[max_inliers]
    H_final = calcular_homografia_dlt(inlier_pts2, inlier_pts1)

    return H_final, max_inliers.tolist()


def calcular_homografia_dlt(p1, p2):
    A = []
    for i in range(len(p1)):
        # Cada punto genera dos ecuaciones (8 en total)
        x, y = p1[i]
        xp, yp = p2[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    # Resolvemos el sistema de ecuaciones
    A = np.array(A)
    _, _, V = np.linalg.svd(A)

    # La última fila de V es el mejor ajuste
    H = V[-1].reshape(3, 3)

    # Normalizamos el resultado (cualquier valor escalar es equitativo)
    return H / H[2, 2]

def dibujar_inliers(img1, kp1, img2, kp2, matches, inliers, nombre='inliers'):
    inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
    resultado = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=2)
    cv2.imwrite(f"results/imagenes_inliers/{nombre}.png", resultado)



def crear_panorama(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Convertir imágenes a BGRA (añadir canal alfa)
    img1_rgba = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img2_rgba = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    # Transformar esquinas de img2
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

    # Calcular dimensiones del panorama
    corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), transformed_corners_img2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Crear el lienzo y pintar img2 en él
    panorama = cv2.warpPerspective(img2_rgba, H_translation @ H, (xmax - xmin, ymax - ymin))

    # Superponer img1 respetando la transparencia
    overlay = np.zeros_like(panorama)
    overlay[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img1_rgba

    # Combinar usando el canal alfa
    alpha_overlay = (overlay[:, :, 3] / 255.0)
    alpha_panorama = (panorama[:, :, 3] / 255.0)

    combined = np.zeros_like(panorama)

    for c in range(3):
        combined[:, :, c] = (
            overlay[:, :, c] * alpha_overlay +
            panorama[:, :, c] * (1 - alpha_overlay)
        ).astype(np.uint8)

    # Actualizar el canal alfa del panorama combinado
    combined[:, :, 3] = np.clip(overlay[:, :, 3] + panorama[:, :, 3], 0, 255)

    return combined


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


