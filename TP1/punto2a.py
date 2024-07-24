import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

ground_truth_df = pd.read_csv("mnyo_ground_truth.csv", sep=" ", header=None, names=["x1", "x2"])
mediciones_1_df = pd.read_csv("mnyo_mediciones.csv", sep=" ", header=None, names=["x1", "x2"])

splines3_x1_vehic1 = CubicSpline(mediciones_1_df.index, mediciones_1_df["x1"]) # uso los índices de las filas del DataFrame como los puntos de referencia para la interpolación
splines3_x2_vehic1 = CubicSpline(mediciones_1_df.index, mediciones_1_df["x2"])

puntos_a_evaluar_v1 = np.linspace(mediciones_1_df.index.min(), mediciones_1_df.index.max(), 1000)

def graficar_trayectoria_splines_3():
    plt.figure(figsize=(12, 5))

    plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Ground Truth', linestyle='-.', color='black')

    plt.scatter(mediciones_1_df["x1"], mediciones_1_df["x2"], label="Mediciones", color='orange')
    plt.plot(splines3_x1_vehic1(puntos_a_evaluar_v1), splines3_x2_vehic1(puntos_a_evaluar_v1), label='Interpolación con Splines Cúbicos', color='red')

    plt.title('Trayedoria Interpolada del primer vehículo')
    plt.xlabel('eje X')
    plt.ylabel('eje Y')

    plt.legend()
    plt.show()

points_to_compare = np.linspace(mediciones_1_df.index.min(), mediciones_1_df.index.max(), 100)
    
def graficar_euclidean_error_abs():
    interp_x1 = splines3_x1_vehic1(points_to_compare)
    interp_x2 = splines3_x2_vehic1(points_to_compare)

    error = np.sqrt((ground_truth_df["x1"] - interp_x1)**2 + (ground_truth_df["x2"] - interp_x2)**2)

    print("Mediana del error absoluto:", error.median())
    print("Máximo del error absoluto:", error.max())
    print("Promedio del error absoluto:", error.mean())

    plt.figure(figsize=(10, 6))
    plt.plot(points_to_compare, error,'o', label = "puntos evaluados", color = 'darkgoldenrod')
    plt.plot(points_to_compare, error, label ="Error absoluto", color = 'orange')
    plt.xlabel('Índice')
    plt.ylabel('Error')
    plt.title("$Error$ $absoluto$ de trayectoria interpolada con Splines Cúbicos contra el ground truth")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    graficar_trayectoria_splines_3()
    graficar_euclidean_error_abs()
    
if __name__ == "__main__":
    main()