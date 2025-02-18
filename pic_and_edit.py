import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

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

        # Mostrar ambas imágenes y los histogramas acumulativos
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Ecualizada')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.plot(acum_hist, color='gray')
        plt.title('Histograma Acumulativo Original')
        plt.xlim([0, 256])

        plt.subplot(2, 2, 4)
        plt.plot(acum_hist2, color='gray')
        plt.title('Histograma Acumulativo Ecualizado')
        plt.xlim([0, 256])

        plt.tight_layout()
        plt.show()

    
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

        # Mostrar ambas imágenes y los histogramas acumulativos
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Imagen Ecualizada')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.plot(acum_hist, color='gray')
        plt.title('Histograma Acumulativo Original')
        plt.xlim([0, 256])

        plt.subplot(2, 2, 4)
        plt.plot(acum_hist2, color='gray')
        plt.title('Histograma Acumulativo Ecualizado')
        plt.xlim([0, 256])

        plt.tight_layout()
        plt.show()


# Contraste - Mejora del contraste de la imagen y ecualización de histograma
def improve_contrast_linear(img, gain, bias, color=False):
    ...
    

# Alien - Cambiar el color de la piel a color rojo, verde o azul
def change_skin(img, factor=1.0):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alto, ancho, _ = img_hsv.shape
    print("Dimensiones de la imagen: ", alto, "x", ancho)
    h, s, v = cv2.split(img_hsv)
            


# Póster - Reducir el número de colores presente en la imagen
def reduce_colors(img):
    ...

# Distorsión - Añadir distorsión de barril y de cojín ajustables
def add_distortion(img):
    ...

# Menu de opciones
def select_menu():
    
    img_taken = False
    end = False
    img = None

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
        print("(8) Salir")
        print("----------------------------------------------------")
        option = input("Seleccione una opción: ")

        if option == '1':
            img = take_picture()
            img_taken = True
        elif option == '2':
            path_img = input("Ingrese la ruta de la imagen: ")
            img = load_image("imgs/img2.jpg")
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
                    if color == 's' or color == 'S':
                        img = improve_contrast_linear(img, gain, bias, color=True)
                    else:
                        img = improve_contrast_linear(img, gain, bias)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '5':
            if img_taken:
                color = input("Ingrese el color (rojo, verde o azul): ")
                img = change_skin(img)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '6':
            if img_taken:
                n_colors = int(input("Ingrese el número de colores: "))
                img = reduce_colors(img)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '7':
            if img_taken:
                k1 = float(input("Ingrese el valor de k1: "))
                k2 = float(input("Ingrese el valor de k2: "))
                img = add_distortion(img)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '8':
            end = True
        else:
            print("Opción no válida")

# Main
if __name__ == '__main__':
    select_menu()


