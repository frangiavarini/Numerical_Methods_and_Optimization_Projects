from TP1.punto1a import equispaced_points, chebyshev_points
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import numpy as np

def fb(x1, x2):
    return 0.75 * np.exp(-((10 * x1 - 2)**2 / 4) - ((9 * x2 - 2)**2 / 4)) + \
           0.65 * np.exp(-((9 * x1 + 1)**2 / 9) - ((10 * x2 + 1)**2 / 2)) + \
           0.55 * np.exp(-((9 * x1 - 6)**2 / 4) - ((9 * x2 - 3)**2 / 4)) - \
           0.01 * np.exp(-((9 * x1 - 7)**2 / 4) - ((9 * x2 - 3)**2 / 4))
xb_min = -1
xb_max = 1

x1_grid = np.linspace(xb_min, xb_max, 100)
x2_grid = np.linspace(xb_min, xb_max, 100)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

def definir_puntos(num_points):
    x1_equispaced_fb = equispaced_points(xb_min, xb_max, num_points)
    x2_equispaced_fb = equispaced_points(xb_min, xb_max, num_points)

    x1_nonequispaced_fb = np.sort(chebyshev_points(xb_min, xb_max, num_points))
    x2_nonequispaced_fb = np.sort(chebyshev_points(xb_min, xb_max, num_points))
   
    X1_equigrid, X2_equigrid = np.meshgrid(x1_equispaced_fb, x2_equispaced_fb)
    z_equispaced_fb = fb(X1_equigrid, X2_equigrid)

    X1_nonequigrid, X2_nonequigrid = np.meshgrid(x1_nonequispaced_fb, x2_nonequispaced_fb)
    z_nonequispaced_fb = fb(X1_nonequigrid, X2_nonequigrid)
    
    return x1_equispaced_fb, x2_equispaced_fb, x1_nonequispaced_fb, x2_nonequispaced_fb, z_equispaced_fb, z_nonequispaced_fb
    
def splines_cubicos(num_points):
    x1_equispaced_fb, x2_equispaced_fb, x1_nonequispaced_fb, x2_nonequispaced_fb, z_equispaced_fb, z_nonequispaced_fb = definir_puntos(num_points)

    spline_equispaced_fb = RectBivariateSpline (x1_equispaced_fb, x2_equispaced_fb, z_equispaced_fb, kx = 3, ky = 3)
    spline_nonequispaced_fb = RectBivariateSpline (x1_nonequispaced_fb, x2_nonequispaced_fb, z_nonequispaced_fb, kx = 3, ky = 3)

    Y_interp_equispaced_fb = spline_equispaced_fb(x1_grid, x2_grid)
    Y_interp_nonequispaced_fb = spline_nonequispaced_fb(x1_grid, x2_grid)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})

    axes[0].plot_surface(X1_grid, X2_grid, Y_interp_equispaced_fb, cmap='viridis', alpha=0.8)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title(f"Interpolación de $f_b(x_1, x_2)$ con Splines Cúbicos \n(P. Equiespaciados: {num_points})")

    axes[1].plot_surface(X1_grid, X2_grid, Y_interp_nonequispaced_fb, cmap='viridis', alpha=0.8)
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_title(f"Interpolación de $f_b(x_1, x_2)$ con Splines Cúbicos \n(P. No Equiespaciados: {num_points})")

    axes[0].plot_wireframe(X1_grid, X2_grid, fb(X1_grid, X2_grid), alpha=0.3, color= "navy")
    axes[1].plot_wireframe(X1_grid, X2_grid, fb(X1_grid, X2_grid), alpha=0.3, color= "navy")
    
    plt.tight_layout()
    plt.show()

def graficar_error_por_nodos(nodes_max, func_text, to_start, kx=None, ky=None):
    Z_real = fb(X1_grid, X2_grid)

    error_equiespaced_median = []
    error_nonequispaced_median = []
    error_equiespaced_max = []
    error_nonequispaced_max = []
    
    q_nodes = range(to_start, nodes_max + 1)
    for nodes in q_nodes:
        x1_eq = np.linspace(xb_min, xb_max, nodes)
        x2_eq = np.linspace(xb_min, xb_max, nodes)
        X1_eq, X2_eq = np.meshgrid(x1_eq, x2_eq)
        Z_interp_eq = RectBivariateSpline(x1_eq, x2_eq, fb(X1_eq, X2_eq), kx=kx, ky=ky)(x1_grid, x2_grid)
        error_eq = np.abs(Z_interp_eq - Z_real)
        error_eq_max= np.max(error_eq)
        error_eq_median = np.median(error_eq)
        error_equiespaced_median.append(error_eq_median)
        error_equiespaced_max.append(error_eq_max)
        
        x1_noneq = np.sort(chebyshev_points(xb_min, xb_max, nodes))
        x2_noneq = np.sort(chebyshev_points(xb_min, xb_max, nodes))
        X1_noneq, X2_noneq = np.meshgrid(x1_noneq, x2_noneq)
        Z_interp_noneq = RectBivariateSpline(x1_noneq, x2_noneq, fb(X1_noneq, X2_noneq), kx=kx, ky=ky)(x1_grid, x2_grid)
        error_noneq = np.abs(Z_interp_noneq - Z_real)
        error_noneq_max= np.max(error_noneq)
        error_noneq_median = np.median(error_noneq)
        error_nonequispaced_median.append(error_noneq_median)
        error_nonequispaced_max.append(error_noneq_max)
        print(nodes, error_eq_median)
       
    plt.figure(figsize=(10, 6))
    plt.plot(q_nodes, error_equiespaced_median, label='$Mediana$ del error absoluto equiespaciado', color = "darkred")
    plt.plot(q_nodes, error_equiespaced_max, label='Máximo del error absoluto equiespaciado', color = "darksalmon")
    plt.plot(q_nodes, error_nonequispaced_median, label='$Mediana$ del error absoluto $\\bf{no}$ equiespaciado', color = "darkgreen")
    plt.plot(q_nodes, error_nonequispaced_max, label='Máximo del error absoluto $\\bf{no}$ equiespaciado', color = "darkseagreen")
    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Error')
    plt.title(f"Error absoluto de interpolación con {func_text} en función de la cantidad de nodos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():   
    splines_cubicos(16)
    graficar_error_por_nodos(40, "Splines Lineales", 2, kx=1, ky=1)  
    graficar_error_por_nodos(40, "Splines Cúbicos", 4, kx=3, ky=3)
    graficar_error_por_nodos(40, "Splines Quínticos", 6, kx=5, ky=5)
    
if __name__ == "__main__":
    main()