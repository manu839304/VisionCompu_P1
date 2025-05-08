import os
import re
import matplotlib.pyplot as plt

def extraer_datos_de_fichero(ruta):
    datos = {
        "metodo": None,
        "nfeatures": None,
        "tiempo_total": None,
        "media_inliers": None,
        "tasa_validos": None,
        "tam_panorama": None,
    }
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if "Método de detección" in linea:
                datos["metodo"] = linea.split(":")[1].strip()
            elif "Número de características" in linea:
                datos["nfeatures"] = int(linea.split(":")[1].strip())
            elif "tiempo_total" in linea:
                datos["tiempo_total"] = float(linea.split(":")[1].strip().split()[0])
            elif "Inliers promedio por par" in linea:
                datos["media_inliers"] = float(linea.split(":")[1].strip())
            elif "Tasa de pares válidos" in linea:
                datos["tasa_validos"] = float(linea.split(":")[1].strip())
            elif "Tamaño del panorama" in linea:
                match = re.search(r"(\d+)\s*x\s*(\d+)", linea)
                if match:
                    datos["tam_panorama"] = (int(match.group(1)), int(match.group(2)))
    return datos

def cargar_todos_los_resultados(carpeta):
    resultados = []
    for archivo in os.listdir(carpeta):
        if archivo.startswith("datos_panorama_res_") and archivo.endswith(".txt"):
            ruta = os.path.join(carpeta, archivo)
            datos = extraer_datos_de_fichero(ruta)
            if datos["metodo"] and datos["nfeatures"]:
                resultados.append(datos)
    return resultados

def graficar_metricas(resultados):
    metodos = list(set(r["metodo"] for r in resultados))
    colores = {"SIFT": "blue", "ORB": "green", "AKAZE": "red"}

    # Ordenar por número de características
    resultados.sort(key=lambda x: (x["metodo"], x["nfeatures"]))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for metodo in metodos:
        subset = [r for r in resultados if r["metodo"] == metodo]
        x = [r["nfeatures"] for r in subset]
        tiempos = [r["tiempo_total"] for r in subset]
        inliers = [r["media_inliers"] for r in subset]
        tasas = [r["tasa_validos"] for r in subset]
        tamanos = [r["tam_panorama"][0] * r["tam_panorama"][1] if r["tam_panorama"] else 0 for r in subset]

        axs[0].plot(x, tiempos, label=metodo, marker='o', color=colores.get(metodo, 'gray'))
        axs[1].plot(x, inliers, label=metodo, marker='o', color=colores.get(metodo, 'gray'))
        axs[2].plot(x, tasas, label=metodo, marker='o', color=colores.get(metodo, 'gray'))
        axs[3].plot(x, tamanos, label=metodo, marker='o', color=colores.get(metodo, 'gray'))

    axs[0].set_title("Tiempo total vs Nº características")
    axs[0].set_xlabel("Nº características")
    axs[0].set_ylabel("Tiempo total (s)")

    axs[1].set_title("Inliers promedio vs Nº características")
    axs[1].set_xlabel("Nº características")
    axs[1].set_ylabel("Inliers promedio")

    axs[2].set_title("Tasa de pares válidos vs Nº características")
    axs[2].set_xlabel("Nº características")
    axs[2].set_ylabel("Tasa de pares válidos")

    axs[3].set_title("Tamaño del panorama vs Nº características")
    axs[3].set_xlabel("Nº características")
    axs[3].set_ylabel("Área del panorama (px²)")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# ===== MAIN =====
if __name__ == "__main__":
    carpeta_resultados = "results/metricas_panoramas"
    resultados = cargar_todos_los_resultados(carpeta_resultados)
    if not resultados:
        print("No se encontraron archivos de resultados.")
    else:
        graficar_metricas(resultados)
