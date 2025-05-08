import os
import re
import subprocess
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
    os.makedirs("results/graficas", exist_ok=True)

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
        tamanos = [r["tam_panorama"][0] * r["tam_panorama"][1] if r["tam_panorama"] else 0 for r in subset]

        axs[0].plot(x, tiempos, label=metodo, marker='o', color=colores.get(metodo, 'gray'))
        axs[1].plot(x, inliers, label=metodo, marker='o', color=colores.get(metodo, 'gray'))
        axs[2].plot(x, tamanos, label=metodo, marker='o', color=colores.get(metodo, 'gray'))

    axs[0].set_title("Tiempo total vs Nº características")
    axs[0].set_xlabel("Nº características")
    axs[0].set_ylabel("Tiempo total (s)")

    axs[1].set_title("Inliers promedio vs Nº características")
    axs[1].set_xlabel("Nº características")
    axs[1].set_ylabel("Inliers promedio")

    axs[2].set_title("Tamaño del panorama vs Nº características")
    axs[2].set_xlabel("Nº características")
    axs[2].set_ylabel("Área del panorama (px²)")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    ruta_grafica = os.path.join("results/graficas", "metricas_panoramas.png")
    plt.savefig(ruta_grafica)
    print(f"Gráficas guardadas en: {ruta_grafica}")
    plt.show()


def ejecutar_pipeline_generacion(carpeta_completa="BuildingScene3", metodos=["SIFT", "ORB", "AKAZE"], nfeatures_list=[500, 1000, 2000]):
    """
    Ejecuta automáticamente panoramas.py para una carpeta específica dentro de VPG/,
    variando método de detección y número de características.

    Parámetros:
        carpeta_completa (str): carpeta con el dataset a usar. ( "BuildingScene3" ,"VPG/*" ...)
        metodos (list): lista de métodos de detección a probar.
        nfeatures_list (list): lista de cantidades de características a usar.
    """

    if not os.path.isdir(carpeta_completa):
        raise FileNotFoundError(f"La carpeta {carpeta_completa} no existe.")

    print(f"Iniciando pipeline para carpeta: {carpeta_completa}")

    for metodo in metodos:
        for nfeatures in nfeatures_list:
            print(f"\nEjecutando: método={metodo}, características={nfeatures}")

            comando = [
                "python", "panorama.py",
                carpeta_completa,
                "--metodo", metodo,
                "--nfeatures", str(nfeatures)
            ]

            try:
                subprocess.run(comando, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error al ejecutar con método={metodo}, nfeatures={nfeatures}: {e}")
            else:
                print(f"Completado: {metodo} con {nfeatures} características")


# ===== MAIN =====
if __name__ == "__main__":

    # Ejecutar el pipeline de generación de panoramas
    ejecutar_pipeline_generacion(carpeta_completa="BuildingScene3", metodos=["SIFT", "ORB", "AKAZE"], nfeatures_list=[500, 700, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 8000, 10000])

    # Ejecutar el análisis de resultados
    carpeta_resultados = "results/metricas_panoramas"
    resultados = cargar_todos_los_resultados(carpeta_resultados)
    if not resultados:
        print("No se encontraron archivos de resultados.")
    else:
        graficar_metricas(resultados)
