import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from matplotlib.pyplot import imshow

_red = (0, 0, 255)
_cyan = (255, 255, 0)

# Funcion para tomar una imagen de la webcam 
def take_picture():
    cap = cv2.VideoCapture(0)
    print("Presione 'q' sobre la ventana de video para tomar la imagen")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Tomar Imagen', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

# Funcion para cargar una imagen desde un archivo
def load_image(path_img):
    img = cv2.imread(path_img)
    return img

# Funcion para calcular el histograma acumulativo de una imagen en escala de grises
def calc_acum_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    acum_hist = np.cumsum(hist)
    return hist, acum_hist

# Muestra en una ventana la imagen, y por consola las propiedades de la matriz que almacena la imagen
def show_image_and_properties(my_img):
    cv2.namedWindow('Imagen', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Imagen', my_img)
    print("Propiedades de la matriz que almacena la imagen")
    print("Son arreglos de numpy: type(my_img)= ", type(my_img))
    print("Filas, columnas y canales: my_img.shape= ", my_img.shape)
    print("Número total de píxeles: my_img.size= ", my_img.size)
    print("Tipo de dato de la imagen: my_img.dtype = ", my_img.dtype)
    cv2.waitKey(0)
    cv2.destroyWindow('Imagen')


def mostrar_imagenes_histogramas(img_orig, img_eq, hist_orig, hist_eq):
    # Mostrar ambas imágenes y los histogramas acumulativos
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Ecualizada')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(hist_orig, color='gray')
    plt.title('Histograma Acumulativo Original')
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.plot(hist_eq, color='gray')
    plt.title('Histograma Acumulativo Ecualizado')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


# Contraste - Mejora del contraste de la imagen y ecualización de histograma
def improve_contrast_acum(img, color=False):

    # Si la imagen es en RGB
    if color:

        # Convertimos la imagen a HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        alto, ancho, _ = img_hsv.shape
        print("Dimensiones de la imagen: ", alto, "x", ancho)

        # Obtenemos el histograma
        num_pixels = ancho * alto
        suma_pixels = 0
        equalized_values = []

        h, s, v = cv2.split(img_hsv)
        hist, acum_hist = calc_acum_hist(v)

        # Calculamos la ecualización de histograma
        for valor, frecuencia in enumerate(hist):
            suma_pixels += frecuencia
            equalized_value = round((suma_pixels * 255) / num_pixels)
            equalized_values.append(equalized_value)
        
        # Construimos la imagen ecualizada
        for y in range(alto):
            for x in range(ancho):
                value = v[y, x]
                v[y, x] = equalized_values[value]
        
        img_hsv = cv2.merge((h, s, v))
        img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        hist2, acum_hist2 = calc_acum_hist(v)

        mostrar_imagenes_histogramas(img, img_eq, acum_hist, acum_hist2)


    
    # Si la imagen es en escala de grises
    else:

        # Cargamos la imagen en escala de grises
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alto, ancho = img_gray.shape
        print("Dimensiones de la imagen: ", alto, "x", ancho)
        
        # Obtenemos el histograma
        num_pixels = ancho * alto
        suma_pixels = 0
        equalized_values = []
        
        hist, acum_hist = calc_acum_hist(img_gray)

        # Calculamos la ecualización de histograma
        for valor, frecuencia in enumerate(hist):
            suma_pixels += frecuencia
            equalized_value = round((suma_pixels * 255) / num_pixels)
            equalized_values.append(equalized_value)

        # Construimos la imagen ecualizada
        for y in range(alto):
            for x in range(ancho):
                value = img_gray[y, x]
                img_gray[y, x] = equalized_values[value]

        hist2, acum_hist2 = calc_acum_hist(img_gray)

        mostrar_imagenes_histogramas(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_gray, acum_hist, acum_hist2)


# Contraste - Mejora del contraste de la imagen y ecualización de histograma
def improve_contrast_linear(img, gain, bias, color=False):
    ...
    

# Alien - Cambiar el color de la piel a color rojo, verde o azul
def change_skin(img, color, factor=1.0):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alto, ancho, _ = img_hsv.shape
    print("Dimensiones de la imagen: ", alto, "x", ancho)
    h, s, v = cv2.split(img_hsv)

    # Recorremos la imagen y pintamos el color piel
    for y in range(alto):
        for x in range(ancho):
            hue = h[y, x]
            sat = s[y, x]
            if (hue <= 15 or hue >= 240)  and sat >= 40 and sat <= 250:
                if color == 'r' or color == 'R':
                    h[y,x] = 0
                elif color == 'g' or color == 'G':
                    h[y,x] = 55
                elif color == 'b' or color == 'B':
                    h[y,x] = 100
                else:
                    return img

                if s[y,x] <= 155:
                    s[y,x] += 100
                else:
                    s[y,x] = 255


    img_hsv = cv2.merge((h, s, v))
    img_pintada = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('pintada', img_pintada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_pintada

def camara_termica(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET) 
    cv2.imshow('thermal', thermal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pixel_art(img, pixel_size=10):

    height, width = img.shape[:2]
    
    # imagen en blanco con las mismas dimensiones
    pixelated_img = np.zeros_like(img)

    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            # bloque de la imagen original
            block = img[y:y+pixel_size, x:x+pixel_size]
            
            avg_color = np.mean(block, axis=(0, 1))  
            
            # bloque en la nueva imagen con el color promedio
            pixelated_img[y:y+pixel_size, x:x+pixel_size] = avg_color

    cv2.imshow('Pixel Art', pixelated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import random
import numpy as np

def glitch(img, intensity=10, color_shift=True):
    height, width, channels = img.shape
    glitched_img = img.copy()

    for i in range(intensity):
        y = random.randint(0, height - 1)
        # desplazamiento aleatorio en el rango de -20 a 20
        shift = random.randint(-20, 20)
        
        # desplazamiento en la fila seleccionada (en el eje horizontal)
        glitched_img[y] = np.roll(img[y], shift, axis=0)

        # efecto de cambio de color
        if color_shift:
            for c in range(3):
                if random.random() > 0.7:
                    # desplazamento del canal horizontalmente
                    shift_channel = random.randint(-5, 5)
                    glitched_img[:, :, c] = np.roll(glitched_img[:, :, c], shift_channel, axis=1) 

        #  ruido aleatorio 
        if random.random() > 0.8:
            # coordenada aleatoria en el eje x
            x = random.randint(0, width - 1)
            glitched_img[y, x] = [random.randint(0, 255) for _ in range(3)]

    cv2.imshow('glitch', glitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def anaglifo(img, shifts=[(-5, 0), (5, 0)], colors=[_red, _cyan]):

    height, width, _ = img.shape
    anaglyph_img = np.zeros_like(img, dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cada color con su desplazamiento
    for (dx, dy), color in zip(shifts, colors):
        # opia de la imagen en escala de grises
        shifted_img = np.zeros_like(img, dtype=np.uint8)

        M = np.float32([[1, 0, dx], [0, 1, dy]]) # M es la matriz de transformación (traslación)
        shifted_gray = cv2.warpAffine(gray, M, (width, height)) # traslación de la imagen en escala de grises
        
        for c, value in enumerate(color):
            # multiplicación de la imagen en escala de grises por el color
            shifted_img[:, :, c] = (shifted_gray * (value / 255)).astype(np.uint8)

        # combinacion de los colores en la imagen final
        anaglyph_img = cv2.add(anaglyph_img, shifted_img)
    
    cv2.imshow('anaglyph', anaglyph_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Póster - Reducir el número de colores presente en la imagen
def reduce_colors(img, kmeans, n_colors):
    if kmeans:
        # Algoritmo basado en https://pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
        (h, w) = img.shape[:2]

        # Pasamos la imagen a LAB porque los colores se perciben de forma más uniforme
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_image = lab_image.reshape((h * w, 3))

        # Organizamos los colores en <n_colores> clusters
        clt = MiniBatchKMeans(n_clusters=n_colors)
        labels = clt.fit_predict(lab_image)

        # Obtiene los colores promedio de cada cluster y lo aplica a todos los colores del cluster
        cluster_centers = clt.cluster_centers_.astype("uint8")[labels]


        cluster_centers = cluster_centers.reshape((h, w, 3))
        posterized = cv2.cvtColor(cluster_centers, cv2.COLOR_LAB2BGR)

        cv2.imshow("posterized", posterized)
        cv2.waitKey(0)

        return posterized

    else:
        # Agrupamos los valores R, G y B en <n_colors> grupos
        # y cambiamos todos los valores de un grupo al valor más bajo del mismo
        factor = 255 // n_colors
        posterized = (img // factor) * factor

        cv2.imshow('posterized', posterized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return posterized

def distorsion_function(x, y, cx, cy, k1, k2):

    # Centramos las coordenadas y las normalizamos
    x_n = (x - cx) / cx
    y_n = (y - cy) / cy

    # Calculamos la distancia radial normalizada desde el centro
    r = np.sqrt(x_n**2 + y_n**2)

    # Aplicamos el modelo de distorsión radial de Brown-Conrady
    x_p = x_n * (1 + k1 * r**2 + k2 * r**4)
    y_p = y_n * (1 + k1 * r**2 + k2 * r**4)

    # Convertimos los valores transformados al sistema de la imagen original
    new_x = int(cx + x_p * cx)
    new_y = int(cy + y_p * cy)

    return new_x, new_y

# Distorsión - Añadir distorsión de barril y de cojín ajustables
def add_distortion(img, k1, k2):
    alto, ancho, c = img.shape
    cx, cy = ancho // 2, alto // 2  # Centro de la imagen
    distorted_img = np.zeros_like(img)

    for y in range(alto):
        for x in range(ancho):
            # Calculamos la posición original de cada píxel en la imagen distorsionada
            x_p, y_p = distorsion_function(x, y, cx, cy, k1, k2)

            if 0 <= x_p < ancho and 0 <= y_p < alto:
                # Aplicamos el pixel calculado original en el pixel distorsionado
                distorted_img[x, y] = img[y_p, x_p]

    cv2.imshow('distorted', distorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return distorted_img

# Menu de opciones
def select_menu():
    
    img_taken = False
    img = None
    end = False

    while not end:
        print("----------------------------------------------------")
        print("PIC AND EDIT")
        print("----------------------------------------------------")
        print("(1) Tomar una imagen desde la webcam")
        print("(2) Cargar una imagen desde un archivo")
        print("(3) Mostrar la imagen original")
        print("(4) Mejorar el contraste")
        print("(5) Cambiar el color de la piel")
        print("(6) Reducir el número de colores")
        print("(7) Añadir distorsión")
        print("(8) Efectos adicionales")
        print("(9) Salir")
        print("----------------------------------------------------")
        option = input("Seleccione una opción: ")

        if option == '1':
            img = take_picture()
            img_taken = True
        elif option == '2':
            path_img = input("Ingrese la ruta de la imagen: ")
            img = load_image(path_img)
            img_taken = True
        elif option == '3':
            if img_taken:
                show_image_and_properties(img)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '4':
            if img_taken:
                acum_or_linear = input("¿Desea realizar la ecualización de histograma acumulativo? (s/n): ")
                color = input("¿Quiere la imagen a color? (s/n): ")
                if acum_or_linear == 's' or acum_or_linear == 'S':
                    if color == 's' or color == 'S':
                        img = improve_contrast_acum(img, color=True)
                    else:
                        img = improve_contrast_acum(img)
                else:
                    gain = input("Introduzca el contraste (gain): ")
                    bias = input("Introduzca el brillo (bias): ")

                    if color == 's' or color == 'S':
                        img = improve_contrast_linear(img, gain, bias, color=True)
                    else:
                        img = improve_contrast_linear(img, gain, bias)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '5':
            if img_taken:
                color = input("Ingrese el color (r/g/b): ")
                img = change_skin(img, color)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '6':
            if img_taken:
                kmeans = input("¿Desea usar clustering k-means? (s/n): ")
                n_colors = 0
                if kmeans == 's' or kmeans == 'S':
                    n_colors = int(input("Ingrese el número de colores:"))
                    kmeans = True
                else:
                    n_colors = int(input("Ingrese el nivel de posterizado:"))
                    kmeans = False

                img = reduce_colors(img, kmeans, n_colors)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '7':
            if img_taken:
                k1 = float(input("Ingrese el valor de k1: "))
                k2 = float(input("Ingrese el valor de k2: "))
                img = add_distortion(img, k1, k2)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '8':
            if img_taken:
                print("----------------------------------------------------")
                print("(1) Camara Termica")
                print("(2) Pixel Art")
                print("(3) Glitch")
                print("(4) Anaglifo")
                print("----------------------------------------------------")
                sub_option = input("Seleccione una opción: ")

                if sub_option == '1':
                    camara_termica(img)
                elif sub_option == '2':
                    num_pixels = int(input("Ingrese el tamaño de los pixeles: "))
                    pixel_art(img, num_pixels)
                elif sub_option == '3':
                    intensity = int(input("Ingrese la intensidad del efecto: "))
                    color_shift = input("¿Desea aplicar cambio de color? (s/n): ")
                    if color_shift == 's' or color_shift == 'S':
                        color_shift = True
                    else:
                        color_shift = False
                    glitch(img, intensity, color_shift)
                elif sub_option == '4':
                    shifts_x1_y1 = input("Ingrese el desplazamiento para el primer color (x, y): ")
                    shifts_x2_y2 = input("Ingrese el desplazamiento para el segundo color (x, y): ")
                    # Si detectamos que no tiene valores, asignamos unos por defecto
                    if not shifts_x1_y1 or not shifts_x2_y2:
                        shifts_x1_y1 = "-10,0"
                        shifts_x2_y2 = "10,0"
                    shifts = [(int(shifts_x1_y1.split(',')[0]),
                               int(shifts_x1_y1.split(',')[1])), 
                              (int(shifts_x2_y2.split(',')[0]), 
                               int(shifts_x2_y2.split(',')[1]))]
                    colors = [_red, _cyan] 
                    anaglifo(img, shifts, colors)
                else:
                    print("Opción no válida")
            else:
                print("\n## Primero debe tomar una imagen ##\n")        
        elif option == '9':
            end = True
        else:
            print("Opción no válida")

# Main
if __name__ == '__main__':
    select_menu()


