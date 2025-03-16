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

# punto de fuga utilizando transformada de Hough (votando solo en la línea del horizonte)
def detectar_punto_fuga(img, operator, threshold=400, paso=1):
    """
      img       : imagen en escala de grises
      operator  : [1,2,3] segun el operador usado en los gradientes
      threshold : valor umbral para permitir voto (en funcion del modulo del gradiente)
      paso      : discretizacion para los candidatos en la línea del horizonte (valor en pixeles)
    """
    alto, ancho = img.shape
    grad_h = calcular_grad_horizontal(img, operator, True)
    grad_v = calcular_grad_vertical(img, operator, True)
    
    # módulo y orientacion en radianes
    mag = np.sqrt(grad_h**2 + grad_v**2)
    orientation = np.arctan2(grad_v, grad_h)
    
    # linea del horizonte 
    y_h = alto // 2
    
    # bins de tamaño 'paso', si se discretiza la linea del hroizonte
    n_bins = ancho // paso
    acum = np.zeros(n_bins, dtype=np.float32)
    
    epsilon = 1e-6  #  para evitar division por cero
    for y in range(alto):
        for x in range(ancho):
            # solo se vota si el modulo es superior al umbral
            if mag[y, x] > threshold:
                theta = orientation[y, x]
                tan_theta = np.tan(theta)
                if abs(tan_theta) < epsilon:
                    continue
                # calculamos el candidato en x que se obtiene al proyectar la linea.
                # y_h = y + tan(theta)*(x_candidate - x)  ==>  x_candidate = x + (y_h - y)/tan(theta)
                x_candidate = x + (y_h - y) / tan_theta
                if 0 <= x_candidate < ancho:
                    # Discretizamos
                    ix = int(round(x_candidate / paso))
                    if ix >= n_bins:
                        ix = n_bins - 1
                    # sumamos un voto al bin correspondiente
                    acum[ix] += mag[y, x]
    
    # el candidato con mayor acumulacion será el punto de fuga
    best_bin = int(np.argmax(acum))
    # pasamos a coordenadas de la imagen
    x_v = int(round(best_bin * paso + paso/2))
    
    print("Punto de fuga encontrado en (x, y):", (x_v, y_h))
    print("Votos máximos acumulados:", acum[best_bin])
    
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(img_color, (x_v, y_h), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=20, thickness=2)
    cv2.imshow('Punto de Fuga Detectado', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
                print("\n## Primero debe tomar o cargar una imagen ##\n")

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
                print("\n## Primero debe tomar o cargar una imagen ##\n")
 
        elif option == '5':
            if img_taken:
                print("----------------------------------------------------")
                print("(1) Sobel")
                print("(2) Scharr")
                print("(3) Canny")
                print("----------------------------------------------------")
                operator = input("Seleccione un operador para los gradientes: ")
                try:
                    threshold = float(input("Ingrese el umbral para el módulo del gradiente: "))
                except Exception as e:
                    print("Valor de umbral incorrecto, se usará 400")
                    threshold = 400
                # se puede ajustar paso segun el nivel de discretización que queramos (paso=1 es cada pixel)
                detectar_punto_fuga(img, operator, threshold, paso=1)
            else:
                print("\n## Primero debe tomar o cargar una imagen ##\n")

        elif option == '6':
            end = True
        else:
            print("Opción no válida")

# Main
if __name__ == '__main__':
    select_menu()