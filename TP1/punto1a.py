import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

def fa(x):
    return 0.3 ** abs(abs((x) ))* np.sin(4 * x) - np.tanh(2 * x) + 2

xa_min = -4
xa_max = 4

def equispaced_points(a, b, num_points):
    return np.linspace(a, b, num_points)

# criterio para elegir puntos no equiespaciados
def chebyshev_points(a, b, num_points):
    return (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(1, num_points + 1) - 1) * np.pi / (2 * num_points))

def generate_midpoints(lst):
        midpoints = []
        for i in range(len(lst) - 1):
            midpoint = (lst[i] + lst[i+1]) / 2.0
            midpoints.append(midpoint)
        return np.array(midpoints)

def definir_puntos(num_points):
    x_equispaced_fa = equispaced_points(xa_min, xa_max, num_points)
    x_nonequispaced_fa = chebyshev_points(xa_min, xa_max, num_points)

    y_equispaced_fa = fa(x_equispaced_fa)
    y_nonequispaced_fa = fa(x_nonequispaced_fa)
    
    return x_equispaced_fa, x_nonequispaced_fa, y_equispaced_fa, y_nonequispaced_fa
    
def graficar_interpol_ambos_puntos(f_interpol_equispaced, f_interpol_nonequispaced, x_equispaced_fa, y_equispaced_fa, x_nonequispaced_fa, y_nonequispaced_fa, x_compare_equipoints_fa, x_compare_nonequipoints_fa, method, q_points):
    plt.figure(figsize=(16, 6))
    
    points_to_study_function = equispaced_points(-3.97, 3.97, 150)
    points_to_study_equifunction = equispaced_points(-4, 4, 150)
    
    plt.plot(points_to_study_function, fa(points_to_study_function), label='$f_a(x)$', linestyle='--', color='black')  # Graficar la función fa(x)
    plt.plot(x_equispaced_fa, y_equispaced_fa, 'o', label='$Puntos de Colocación$ (Equispaciado)', color = 'blue')
    plt.plot(points_to_study_equifunction, f_interpol_equispaced(points_to_study_equifunction), label='$Interpolación$ (Equispaciado)', color = 'green')
    plt.plot(x_nonequispaced_fa, y_nonequispaced_fa, 'o', label='$Puntos de Colocación$ (No equiespaciado)', color = 'orange')
    plt.plot(points_to_study_function, f_interpol_nonequispaced(points_to_study_function), label='$Interpolación$ (No equiespaciado)', color = 'red')
    plt.xlabel('$x$')
    plt.ylabel('$f_a(x)$')
    plt.title(f"Interpolación de $f_a(x)$ con {method} con {q_points} Puntos de Colocación")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def graficar_error(f_interpol_equispaced, f_interpol_nonequispaced, x_compare_equipoints_fa, x_compare_nonequipoints_fa, method, q_points):
    error_equispaced_w_midpoints = np.abs(fa(x_compare_equipoints_fa) - f_interpol_equispaced(x_compare_equipoints_fa))
    error_nonequispaced_w_midpoints = np.abs(fa(x_compare_nonequipoints_fa) - f_interpol_nonequispaced(x_compare_nonequipoints_fa))

    median_error_equispaced = np.median(error_equispaced_w_midpoints)
    median_error_nonequispaced = np.median(error_nonequispaced_w_midpoints)

    legend_equispaced = f'$Error$ $absoluto$ (Equispaciado) - \n$Mediana:$ {median_error_equispaced:.5f}'
    legend_nonequispaced = f'$Error$ $absoluto$ (No equiespaciado) - \n$Mediana:$ {median_error_nonequispaced:.5f}'
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_compare_equipoints_fa, error_equispaced_w_midpoints, 'o')
    plt.plot(x_compare_nonequipoints_fa, error_nonequispaced_w_midpoints, 'o')
    plt.plot(x_compare_equipoints_fa, error_equispaced_w_midpoints, label='$Error$ $absoluto$ (Equispaciado)')
    plt.plot(x_compare_nonequipoints_fa, error_nonequispaced_w_midpoints, label='$Error$ $absoluto$ (No equiespaciado)')
    plt.xlabel('$x$')
    plt.ylabel('$Error$')
    plt.title(f"Error absoluto de Interpolación con {method} de $f_a(x)$ con {q_points} puntos")
    plt.legend([legend_equispaced, legend_nonequispaced], fontsize='small')
    plt.grid(True)
    plt.show()

def interpolacion_lineal(x, x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m * (x - x1) + y1

def linear_interpol(num_points):
    
    x_equispaced_fa, x_nonequispaced_fa, y_equispaced_fa, y_nonequispaced_fa = definir_puntos(num_points)

    fa_linear_equispaced = interp1d(x_equispaced_fa, y_equispaced_fa)
    fa_linear_nonequispaced = interp1d(x_nonequispaced_fa, y_nonequispaced_fa)

    x_compare_equipoints_fa = generate_midpoints(x_equispaced_fa)
    x_compare_nonequipoints_fa = generate_midpoints(x_nonequispaced_fa)
    
    graficar_interpol_ambos_puntos(fa_linear_equispaced, fa_linear_nonequispaced, x_equispaced_fa, y_equispaced_fa, x_nonequispaced_fa, y_nonequispaced_fa, x_compare_equipoints_fa, x_compare_nonequipoints_fa, "Lineal", num_points)

def lagrange_interpol(num_points):
            
    x_equispaced_fa, x_nonequispaced_fa, y_equispaced_fa, y_nonequispaced_fa = definir_puntos(num_points)

    fa_lagrange_equispaced = lagrange(x_equispaced_fa, y_equispaced_fa)
    fa_lagrange_nonequispaced = lagrange(x_nonequispaced_fa, y_nonequispaced_fa)

    x_compare_equipoints_fa = generate_midpoints(x_equispaced_fa)
    x_compare_nonequipoints_fa = generate_midpoints(x_nonequispaced_fa)
    
    graficar_interpol_ambos_puntos(fa_lagrange_equispaced, fa_lagrange_nonequispaced, x_equispaced_fa, y_equispaced_fa, x_nonequispaced_fa, y_nonequispaced_fa, x_compare_equipoints_fa, x_compare_nonequipoints_fa, "Lagrange", num_points)

def splines_cubic_interpol(num_points):
    
    x_equispaced_fa, x_nonequispaced_fa, y_equispaced_fa, y_nonequispaced_fa = definir_puntos(num_points)

    spline_equispaced_fa = CubicSpline(x_equispaced_fa, y_equispaced_fa)

    sorted_few_indices = np.argsort(x_nonequispaced_fa)
    x_nonequispaced_fa_sorted = x_nonequispaced_fa[sorted_few_indices]
    y_nonequispaced_fa_sorted = y_nonequispaced_fa[sorted_few_indices]
    
    x_compare_equipoints_fa = generate_midpoints(x_equispaced_fa)
    x_compare_nonequipoints_fa = generate_midpoints(x_nonequispaced_fa_sorted)

    spline_nonequispaced_fa = CubicSpline(x_nonequispaced_fa_sorted, y_nonequispaced_fa_sorted)
    
    graficar_interpol_ambos_puntos(spline_equispaced_fa, spline_nonequispaced_fa, x_equispaced_fa, y_equispaced_fa, x_nonequispaced_fa, y_nonequispaced_fa, x_compare_equipoints_fa, x_compare_nonequipoints_fa, "Splines Cúbicos", num_points)

def graficar_error_por_nodos(nodes_q, func, func_text):
    x = np.linspace(xa_min, xa_max, 100)
    
    y = fa(x)
    error_equiespaced_median = []
    error_nonequispaced_median = []
    error_equiespaced_max = []
    error_nonequispaced_max = []
    
    nodes = [x for x in range(2, nodes_q+1)]
    for node in nodes:
        x_equispaced = np.linspace(xa_min, xa_max, node)
        x_nonequispaced = np.sort(chebyshev_points(xa_min, xa_max, node))
        
        if func == interp1d:
            x_min = max(xa_min, min(x_nonequispaced))
            x_max = min(xa_max, max(x_nonequispaced))
            x = np.linspace(x_min, x_max, 100)
        
        z_equispaced = fa(x_equispaced)
        z_nonequispaced = fa(x_nonequispaced)
        
        y_interp_equispaced = func(x_equispaced, z_equispaced) (x)
        y_interp_nonequispaced = func(x_nonequispaced, z_nonequispaced) (x)
        
        error_equispaced = np.abs(y - y_interp_equispaced)
        error_nonequispaced = np.abs(y - y_interp_nonequispaced)
        
        error_equiespaced_median.append(np.median(error_equispaced))
        error_nonequispaced_median.append(np.median(error_nonequispaced))
        
        error_equiespaced_max.append(np.max(error_equispaced))
        error_nonequispaced_max.append(np.max(error_nonequispaced))
        
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, error_equiespaced_median, label='$Mediana$ del error absoluto equiespaciado', color = "darkred")
    plt.plot(nodes, error_equiespaced_max, label='Máximo del error absoluto equiespaciado', color = "darksalmon")
    plt.plot(nodes, error_nonequispaced_median, label='$Mediana$ del error absoluto $\\bf{no}$ equiespaciado', color = "darkgreen")
    plt.plot(nodes, error_nonequispaced_max, label='Máximo del error absoluto $\\bf{no}$ equiespaciado', color = "darkseagreen")
    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Error')
    plt.title(f"Error absoluto de interpolación con {func_text} en función de la cantidad de nodos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    linear_interpol(36)
    graficar_error_por_nodos(40, interp1d, "Interpolación lineal")
    lagrange_interpol(13)
    lagrange_interpol(20)
    graficar_error_por_nodos(20, lagrange, "Lagrange")
    graficar_error_por_nodos(35, CubicSpline, "Splines Cúbicos")
    splines_cubic_interpol(20)
    
if __name__ == "__main__":
    main()