import pandas as pd
from TP1.punto2a import *
import numpy as np
import matplotlib.pyplot as plt

mediciones_2_df = pd.read_csv("mnyo_mediciones2.csv", sep=" ", header=None, names=["x1", "x2"])

puntos_a_evaluar_v2 = np.linspace(mediciones_2_df.index.min(), mediciones_2_df.index.max(), 1000)

splines3_x1_vehic2 = CubicSpline(mediciones_2_df.index, mediciones_2_df["x1"]) # uso los índices de las filas del DataFrame como los puntos de referencia para la interpolación
splines3_x2_vehic2 = CubicSpline(mediciones_2_df.index, mediciones_2_df["x2"])

x1_vehiculo1, x2_vehiculo1 = mediciones_1_df["x1"], mediciones_1_df["x2"]
x1_vehiculo2, x2_vehiculo2 = mediciones_2_df["x1"], mediciones_2_df["x2"]

def interseccion_trayec_splines3(t, t2):
    x1_trayec_vehic1, x2_trayec_vehic1 = splines3_x1_vehic1(t), splines3_x2_vehic1(t)
    x1_trayec_vehic2, x2_trayec_vehic2 = splines3_x1_vehic2(t2), splines3_x2_vehic2(t2)
    return (x1_trayec_vehic1 - x1_trayec_vehic2, x2_trayec_vehic1 - x2_trayec_vehic2)

def newton_2d(f, P0, P1, tol=1e-6, max_iter=100):
    P = np.array([P0, P1])
    n = 0
    for _ in range(max_iter):
        f_val = f(*P)
        n+=1
        if np.linalg.norm(f_val) < tol:
            break
        J = np.array([[f(P[0]+tol, P[1])[0]-f_val[0], f(P[0], P[1]+tol)[0]-f_val[0]],
                      [f(P[0]+tol, P[1])[1]-f_val[1], f(P[0], P[1]+tol)[1]-f_val[1]]]) / tol
        delta_P = np.linalg.solve(J, np.array([-f_val[0], -f_val[1]]))
        P = P + delta_P
    print(f"El método convergió en {n} iteraciones")
    return tuple(P)

t0 = mediciones_1_df.index.min()
t1 = mediciones_2_df.index.min()

t_interseccion, t2_interseccion = newton_2d(interseccion_trayec_splines3, t0, t1)

x_interseccion_1, y_interseccion_1 = splines3_x1_vehic1(t_interseccion), splines3_x2_vehic1(t_interseccion)
x_interseccion_2, y_interseccion_2 = splines3_x1_vehic2(t2_interseccion), splines3_x2_vehic2(t2_interseccion)

error_x1 = np.abs(x_interseccion_1 - x_interseccion_2)
error_x2 = np.abs(y_interseccion_1 - y_interseccion_2)


print(f"tiempos de intersección-> vehículo 1:{t_interseccion} y vehículo 2:{t2_interseccion}")
print(f"Coordenadas de la intersección-> vehículo 1:({x_interseccion_1}, {y_interseccion_1}) y vehículo 2:({x_interseccion_2}, {y_interseccion_2})")

def graficar_trayectorias_intersec():
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Ground Truth', linestyle='-.', color='dimgrey')

    plt.scatter(mediciones_1_df["x1"], mediciones_1_df["x2"], label="Mediciones v1", color='yellowgreen', s=25)
    plt.plot(splines3_x1_vehic1(puntos_a_evaluar_v1), splines3_x2_vehic1(puntos_a_evaluar_v1), label='$\sigma_1(t)$ Interpolación con Splines Cúbicos v1', color='teal')

    plt.scatter(mediciones_2_df["x1"], mediciones_2_df["x2"], label="Mediciones v2", color='plum', s=25)
    plt.plot(splines3_x1_vehic2(puntos_a_evaluar_v2), splines3_x2_vehic2(puntos_a_evaluar_v2), label="$\sigma_2(t')$ Interpolation con Splines Cúbicos v2", color='mediumpurple')

    plt.plot(x_interseccion_1, y_interseccion_1, 'x', label="Intersección", markersize=8.5, color='darkred',  markeredgewidth=4)    
    plt.xlabel("Coordenada $x1$")
    plt.ylabel("Coordenada $x2$")
    plt.title("Trayectorias de los 2 Vehículos e intersección")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def graficar_error_entre_coordenadas():
    
    plt.figure(figsize=(7, 6))
    indice = np.arange(2)

    plt.bar(indice, [error_x1, error_x2], 0.4, color=['khaki', 'darkseagreen'])
    plt.xlabel('$Dimensiones$')
    plt.ylabel('$Error$')
    plt.title('$Error$ entre las Coordenadas de la Intersección')
    plt.xticks(indice, ('$x1$', '$x2$'))
    text_offset = 0.0000000000035
    plt.text(0, text_offset, f"Diferencia: {round(error_x1, 15)}", ha='center', va='bottom', fontsize=9)
    plt.text(1, text_offset, f"Diferencia: {round(error_x2, 16)}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

def main():
    graficar_trayectorias_intersec()
    graficar_error_entre_coordenadas()
    
if __name__ == "__main__":
    main()