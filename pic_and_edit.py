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
def improve_contrast(img):
    # Mostrar el histograma de la imagen
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    

# Alien - Cambiar el color de la piel a color rojo, verde o azul
def change_skin(img):
    ...

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
            img = load_image(path_img)
            img_taken = True
        elif option == '3':
            if img_taken:
                show_image_and_properties(img)
            else:
                print("\n## Primero debe tomar una imagen ##\n")
        elif option == '4':
            if img_taken:
                alpha = float(input("Ingrese el valor de alpha: "))
                beta = int(input("Ingrese el valor de beta: "))
                img = improve_contrast(img)
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


