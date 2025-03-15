import cv2 
import numpy as np
from numpy.ma.core import arctan2


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
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    return img

# Muestra en una ventana la imagen, y por consola las propiedades de la matriz que almacena la imagen
def show_image_and_properties(my_img):
    cv2.imshow('Imagen', my_img)
    print("Propiedades de la matriz que almacena la imagen")
    print("Son arreglos de numpy: type(my_img)= ", type(my_img))
    print("Filas, columnas y canales: my_img.shape= ", my_img.shape)
    print("Número total de píxeles: my_img.size= ", my_img.size)
    print("Tipo de dato de la imagen: my_img.dtype = ", my_img.dtype)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_filter(value, sigma = 1):
    return np.exp(-(value ** 2)/(2 * sigma ** 2)) 

def gaussian_derivative(value, sigma = 1):
    return ((-value / (sigma ** 2))) * (np.exp(-(value ** 2)/(2 * sigma ** 2)))


def calcular_grad_horizontal(img, op, ret = False):
    kernel = None
    alto, ancho = img.shape

    # Seleccinamos el kernel
    if op == "1": # Sobel
        kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    elif op == "2": # Scharr
        kernel = np.array([[-3, 0, 3],
                        [-10, 0, 10],
                        [-3, 0, 3]], dtype=np.float32) 

    elif op == "3": # Canny
        
        # kernels separados (horizontal y vertical)
        kernel_h = np.array([[gaussian_derivative(2),
                              gaussian_derivative(1),
                              gaussian_derivative(0),
                              gaussian_derivative(-1),
                              gaussian_derivative(-2)]], dtype=np.float32)
        
        kernel_v = np.array([[[gaussian_filter(2)],
                                [gaussian_filter(1)],
                                [gaussian_filter(0)],
                                [gaussian_filter(-1)],
                                [gaussian_filter(-2)]]], dtype=np.float32)
        
        # Calculamos K, que es la suma de los valores positivos tanto del kernel horizontal como del vertical
        k = sum(v for v in kernel_h[0] if v > 0) + sum(v[0] for v in kernel_v if v[0] > 0)
        
        kernel = (1/k) * (kernel_h * kernel_v)
        kernel = np.array(kernel[0], dtype=np.float32)

    
    else:
        print("Operador no válido")
        return None

    img_gaus = cv2.GaussianBlur(img, (3, 3), 0)
    grad_h = cv2.filter2D(img_gaus, ddepth=cv2.CV_64F, kernel=kernel)


    if ret:
        return grad_h
    
    else:
        array_grad_h = grad_h.flatten()
        print("Valor máximo: ", max(array_grad_h))
        print("Valor mínimo: ", min(array_grad_h))
        showable_grad_h = np.zeros((alto, ancho, 1), dtype=np.uint8)
        for y in range(alto):
            for x in range(ancho):
                showable_grad_h[y,x] = np.clip(grad_h[y,x]/2 + 128, 0, 255).astype(np.uint8) 
        
        cv2.imshow('Gradiente Horizontal', showable_grad_h)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def calcular_grad_vertical(img, op, ret = False):
    kernel = None
    alto, ancho = img.shape

    # Seleccinamos el kernel
    if op == "1": # Sobel
        kernel = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    elif op == "2": # Scharr
        kernel = np.array([[3, 10, 3],
                        [0, 0, 0],
                        [-3, -10, -3]], dtype=np.float32) 

    elif op == "3": # Canny
        
        # kernels separados (horizontal y vertical)
        kernel_h = np.array([[gaussian_filter(-2),
                              gaussian_filter(-1),
                              gaussian_filter(0),
                              gaussian_filter(1),
                              gaussian_filter(2)]], dtype=np.float32)
        
        kernel_v = np.array([[[gaussian_derivative(-2)],
                                [gaussian_derivative(-1)],
                                [gaussian_derivative(0)],
                                [gaussian_derivative(1)],
                                [gaussian_derivative(2)]]], dtype=np.float32)
        
        # Calculamos K, que es la suma de los valores positivos tanto del kernel horizontal como del vertical
        k = sum(v for v in kernel_h[0] if v > 0) + sum(v[0] for v in kernel_v if v[0] > 0)
        
        kernel = (1/k) * (kernel_h * kernel_v)
        kernel = np.array(kernel[0], dtype=np.float32)

    
    else:
        print("Operador no válido")
        return None

    img_gaus = cv2.GaussianBlur(img, (3, 3), 0)
    grad_v = cv2.filter2D(img_gaus, ddepth=cv2.CV_64F, kernel=kernel)

    if ret:
        return grad_v
    
    else:
        array_grad_v = grad_v.flatten()
        print("Valor máximo: ", max(array_grad_v))
        print("Valor mínimo: ", min(array_grad_v))
        showable_grad_v = np.zeros((alto, ancho, 1), dtype=np.uint8)
        for y in range(alto):
            for x in range(ancho):
                showable_grad_v[y,x] = np.clip(grad_v[y,x]/2 + 128, 0, 255).astype(np.uint8) 
        
        cv2.imshow('Gradiente Vertical', showable_grad_v)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

def calcular_modulo_grad(img, operator):
    alto, ancho = img.shape

    grad_h = calcular_grad_horizontal(img, operator, True)
    grad_v = calcular_grad_vertical(img, operator, True)

    mod_grad = np.zeros((alto, ancho, 1), dtype=np.float32)
    showable_grad = np.zeros((alto, ancho, 1), dtype=np.uint8)

    for y in range(alto):
        for x in range(ancho):
            mod_grad[y,x] = np.sqrt(grad_h[y,x] ** 2 + grad_v[y,x] ** 2)
            showable_grad[y,x] = np.clip(mod_grad[y,x], 0, 255).astype(np.uint8)

    array_mod_v = mod_grad.flatten()
    print("Valor máximo: ", max(array_mod_v))
    print("Valor mínimo: ", min(array_mod_v))

    cv2.imshow('Gradiente Modulo', showable_grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def calcular_orientacion_grad(img, operator):
    alto, ancho = img.shape

    grad_h = calcular_grad_horizontal(img, operator, True)
    grad_v = calcular_grad_vertical(img, operator, True)

    showable_grad = np.zeros((alto, ancho, 1), dtype=np.uint8)
    dir_grad = np.zeros((alto, ancho, 1), dtype=np.float32)

    for y in range(alto):
        for x in range(ancho):
            atan2 = arctan2(grad_v[y,x], grad_h[y,x])
            atan2_deg = atan2 * (180/np.pi)
            dir_grad[y, x] = atan2_deg
            showable_grad[y, x] = np.clip((atan2_deg / 2 + 90) * (255/180), 0, 255).astype(np.uint8)

    array_dir_v = dir_grad.flatten()
    print("Valor máximo: ", max(array_dir_v))
    print("Valor mínimo: ", min(array_dir_v))

    cv2.imshow('Gradiente Direccion', showable_grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Menú gráfico que permite 
def select_menu():
    img_taken = False
    img = None
    end = False

    while not end:
        print("----------------------------------------------------")
        print("PRACTICA 2")
        print("----------------------------------------------------")
        print("(1) Tomar una imagen desde la webcam")
        print("(2) Cargar una imagen desde un archivo")
        print("(3) Mostrar la imagen original")
        print("(4) Mostrar gradiente")
        print("(5) Detectar punto de fuga")
        print("(6) Salir")
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
                print("----------------------------------------------------")
                print("(1) Sobel")
                print("(2) Scharr")
                print("(3) Canny")
                print("----------------------------------------------------")
                operator = input("Seleccione un operador: ")

                print("----------------------------------------------------")
                print("(1) Gradiente Horizontal")
                print("(2) Gradiente Vertical")
                print("(3) Módulo")
                print("(4) Orientación")
                print("----------------------------------------------------")
                sub_option = input("Seleccione una opción: ")


                if sub_option == '1':
                    calcular_grad_horizontal(img, operator)
                    
                elif sub_option == '2':
                    calcular_grad_vertical(img, operator)
                    
                elif sub_option == '3':
                    calcular_modulo_grad(img, operator)
                    
                elif sub_option == '4':
                    calcular_orientacion_grad(img, operator)
                else:
                    print("Opción no válida")
            else:
                print("\n## Primero debe tomar una imagen ##\n")
 
        elif option == '6':
            end = True
        else:
            print("Opción no válida")

# Main
if __name__ == '__main__':
    select_menu()