import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from punto2 import runge_kutta4_system


def dN_dt_LVE (N, P, r, alpha, K):
    return r * N * ( 1 - N / K ) - alpha * N * P

def dN_dt (N, P, r, alpha):
    return r * N - alpha * N * P

def dP_dt (N, P, beta, q):
    return beta * N * P - q * P

def lotka_volterra_LVE(t, y0, r1, r2, K1, alpha, beta, q):
    N, P = y0
    return np.array([dN_dt_LVE(N, P, r1, alpha, K1), dP_dt(N, P, beta, q)])

def lotka_volterra(t, y0, r1, r2, alpha, beta, q):
    N, P = y0
    return np.array([dN_dt(N, P, r1, alpha), dP_dt(N, P, beta, q)])


def punto_equilibrio_LVE(r, K, alpha, beta, q):
    def f(x):
        return [r * x[0] * (1 - x[0] / K ) - alpha * x[0] * x[1], beta * x[0] * x[1] - q * x[1]]
    
    return fsolve(f, [20, 20])

def punto_equilibrio(r, alpha, beta, q, punto = [100, 200]):
    def f(x):
        return [r * x[0] - alpha * x[0] * x[1], beta * x[0] * x[1] - q * x[1]]
    return fsolve(f, punto)

def isoclinas_y_campo_vectorial(r, alpha, beta, q, title, lim, puntos_iniciales, to_graph = True):
    
    N = np.linspace(0, 500, 100)
    P = np.linspace(0, 1000, 100)

    isocline_p = q / beta
    isocline_n = r / alpha
    
    VN, VP = np.meshgrid(N, P)
    
    dN = dN_dt(VN, VP, r, alpha)
    dP = dP_dt(VN, VP, beta, q)
    
    magnitude = np.sqrt(dN**2 + dP**2)
    
    if to_graph:
        plt.figure()
    
    
    plt.axhline(y= isocline_n, color='limegreen', label='dn/dt = 0 (isoclina de N)')
    plt.axvline(x=isocline_p, color='firebrick', label='dp/dt = 0 (isoclina de P)')
    
    punto_eq = punto_equilibrio(r, alpha, beta, q)
    
    if beta == 0.0009 and q == 0.4:
        punto_eq = [444.4, 20]
        
    if punto_eq[0] == 0 and punto_eq[1] == 0:
        punto_eq = punto_equilibrio(r, alpha, beta, q, [800, -400])
    
    if punto_eq[0] == 0 and punto_eq[1] == 0:
        punto_eq = punto_equilibrio(r, alpha, beta, q, [1.5, 2.5])
    
    plt.plot(punto_eq[0], punto_eq[1], 'o', color='teal', markersize=10, label='Punto de equilibrio')
    
    p0_plotted = False 

    plt.plot([], [], color='darkslategray', label='Aproximación de (N1(t), N2(t)) con RK4 desde distintos p0')
    
    for punto in puntos_iniciales:
        _, y_values = runge_kutta4_system(lotka_volterra, 0, punto, tf, h, r, r, alpha, beta, q)
        plt.plot(y_values[:, 0], y_values[:, 1], color = 'darkslategray')
        if not p0_plotted: 
            plt.plot(punto[0], punto[1], 'o', color='darkslategray', markersize=5, label="p0's")
            p0_plotted = True
        else:
            plt.plot(punto[0], punto[1], 'o', color='darkslategray', markersize=5)
            
    plt.streamplot(N, P, dN, dP, color=magnitude, linewidth=0.5, cmap='CMRmap', arrowstyle='->', arrowsize=0.8)
  
    plt.grid()
    
    plt.xlabel('Población de Presas (N)', fontsize=13)
    plt.ylabel('Población de predadores (P)', fontsize=13)
    plt.title(title, fontsize=20)
    plt.ylim(lim[0][0], lim[0][1])
    plt.xlim(lim[1][0], lim[1][1])
    
    plt.legend(handlelength=0.75, fontsize = 6)
    
    if to_graph:
        plt.show()
        
def isoclinas_y_campo_vectorial_varias(cases, LVE = False):
    plt.figure(figsize=(10, 10))

    for i, case in enumerate(cases.values(), start=1):
        plt.subplot(2, 2, i)
        isoclinas_y_campo_vectorial(case['r'], case['alpha'], case['beta'], case['q'], case['title'], case['lim'], case['puntos_iniciales'], False)
        # if i == 2:
        #     break
    
    plt.suptitle('Isoclinas LV (simple)', fontsize=20)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def isoclinas_y_campo_vectorial_LVE(r, K, alpha, beta, q, title, coef, lim, puntos_iniciales, to_graph = True):
    
    N = np.linspace(0, K, 100)
    P = np.linspace(0, K, 100)

    isocline_p = q / beta
    isocline_n = (r * (1 - N / K)) / alpha
    
    VN, VP = np.meshgrid(N, P)
    
    dN = dN_dt_LVE(VN, VP, r, alpha, K)
    dP = dP_dt(VN, VP, beta, q)
    
    magnitude = np.sqrt(dN**2 + dP**2)
    
    if to_graph:
        plt.figure()
    
    for punto in puntos_iniciales:
        _, y_values = runge_kutta4_system(lotka_volterra_LVE, 0, punto, tf, h, r, r, K, alpha, beta, q)
        plt.plot(y_values[:, 0], y_values[:, 1], color = 'darkslategray')
        plt.plot(punto[0], punto[1], 'o', color='darkslategray', markersize=5)
        
    plt.plot(N, isocline_n, color='limegreen')
    plt.axvline(x=isocline_p, color='firebrick')
    punto_eq = punto_equilibrio_LVE(r, K, alpha, beta, q)
        
    plt.plot(punto_eq[0], punto_eq[1], 'o', color='teal', markersize=10, label='Punto de equilibrio')
    print(punto_eq)

    plt.plot([], [], color='darkslategray', label='Aproximación de (N1(t), N2(t)) \ncon RK4 desde distintos p0')
       
    strm = plt.streamplot(VN, VP, dN, dP, color=magnitude, linewidth=0.3, cmap='CMRmap', arrowstyle='->', arrowsize=0.8)
    plt.grid()
    
    plt.xlabel('Población de Presas (N)', fontsize=13)
    plt.ylabel('Población de predadores (P)', fontsize=13)
    plt.title(title + ': ' + coef, fontsize=20)
    plt.ylim(lim[0][0], lim[0][1])
    plt.xlim(lim[1][0], lim[1][1])
    
    plt.legend(fontsize=8, handlelength=0.75)
    # cbar = plt.colorbar(strm.lines)
    # cbar.set_label(label='Magnitud del campo vectorial', fontsize=12)
    
    if to_graph:
        plt.show()

def isoclinas_y_campo_vectorial_LVE_varias(cases, LVE = False):
    plt.figure(figsize=(10, 10))

    for i, case in enumerate(cases.values(), start=1):
        plt.subplot(2, 2, i)
        isoclinas_y_campo_vectorial_LVE(case['r'], case['K'], case['alpha'], case['beta'], case['q'], case['title'], case['coef'] ,case['lim'], case['puntos_iniciales'], False)
        if i == 2:
            break
        
    plt.subplots_adjust(wspace=0.4)
    plt.show()

def graficar_soluciones_rk_varias(t0, N1_0, N2_0, h, cases):
    plt.figure(figsize=(10, 10))

    for i, case in enumerate(cases.values(), start=1):
       
        if i == 4:
            [N1_0, N2_0] = [100, 50]
        t_values, y_values_LVE = runge_kutta4_system(lotka_volterra_LVE, t0, [N1_0, N2_0], case['tf'], h, case['r'], case['r'], case['K'], case['alpha'], case['beta'], case['q'])
        t_values, y_values = runge_kutta4_system(lotka_volterra, t0, [N1_0, N2_0], case['tf'], h, case['r'], case['r'], case['alpha'], case['beta'], case['q'])
        
        plt.subplot(2, 2, i)        
        plt.plot(t_values, y_values_LVE[:, 0], label='N(t) (modelo LVE)', color = 'peru')
        plt.plot(t_values, y_values[:, 0], label='N(t) (modelo original)', color = 'seagreen')
        plt.plot(t_values, y_values_LVE[:, 1], label='P(t) (modelo LVE)', color = 'darkred')
        plt.plot(t_values, y_values[:, 1], label='P(t) (modelo original)', color = 'dodgerblue')
        plt.xlabel('Tiempo', fontsize=15)
        plt.ylabel('Población', fontsize=15)
        plt.title(case['title'] + ' ('+ case['coef'] + ')', fontsize=16)
        
        if i == 4:
            plt.legend(fontsize = 15)
    
    plt.suptitle('Evolución temporal de las poblaciones', fontsize=20)
    plt.subplots_adjust(hspace=0.55, wspace=0.3)
    plt.show()

def graficar_sol_rk (t0, y0, tf, h, r, K, alpha, beta, q, title):
    t_values, y_values = runge_kutta4_system(lotka_volterra_LVE, t0, y0, tf, h, r, r, K, alpha, beta, q)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values[:, 0], label='presas')
    plt.plot(t_values, y_values[:, 1], label='depredadores')
    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Población')
    plt.title('Dinámica de la población de presas y depredadores (' + title +')')
    plt.legend()
    plt.grid(True)
    plt.show()

def variaciones( t0, y0, tf, h, r, K, alpha, beta, q, to_graph = True, LVE = False):
    
    if to_graph:
        plt.figure(figsize=(10, 6))
        
    if LVE:
        t_values, y_values_LVE = runge_kutta4_system(lotka_volterra_LVE, t0, y0, tf, h, r, r, K, alpha, beta, q)
        N_values_LVE = y_values_LVE[:, 0]
        P_values_LVE = y_values_LVE[:, 1]
        
        dN_values_LVE = dN_dt_LVE(N_values_LVE, P_values_LVE, r, alpha, K)
        dP_values_LVE = dP_dt(N_values_LVE, P_values_LVE, beta, q)
    
        plt.plot(t_values, dN_values_LVE, label='Variación de N(t) en función de t (LVE)', color='peru')
        plt.plot(t_values, dP_values_LVE, label='Variación de P(t) en función de t (LVE)', color='darkred')
        
    else:
        t_values, y_values = runge_kutta4_system(lotka_volterra, t0, y0, tf, h, r, r, alpha, beta, q)
        N_values = y_values[:, 0]
        P_values = y_values[:, 1]
        dN_values = dN_dt(N_values, P_values, r, alpha)
        dP_values = dP_dt(N_values, P_values, beta, q)
      
        plt.plot(t_values, dN_values, label='Variación de N(t) en función de t (LV simple)', color='seagreen')
        plt.plot(t_values, dP_values, label='Variación de P(t) en función de t (LV simple)', color='dodgerblue')
        
    plt.xlabel('Tiempo (t)', fontsize=18)
    plt.ylabel('Variación', fontsize=18)
    
    if LVE:
        plt.title('LVE', fontsize=16)
    
    else: 
        plt.title('LV simple', fontsize=16)
        
    plt.legend(fontsize=12)
    plt.grid(True)
    if to_graph:
        plt.show()
    
def dos_variaciones(t0, y0, tf, h, cases):
    plt.figure()
    plt.subplot(1, 2, 1)
    variaciones(t0, y0, tf, h, cases['a']['r'], cases['a']['K'], cases['a']['alpha'], cases['a']['beta'], cases['a']['q'], False)
    plt.subplot(1, 2, 2)
    variaciones(t0, y0, tf, h, cases['a']['r'], cases['a']['K'], cases['a']['alpha'], cases['a']['beta'], cases['a']['q'], False, True)
    plt.suptitle('Variación de las poblaciones de presas y depredadores: caso a', fontsize = 20)
    plt.show()

def tamaño_poblacional(t0, y0, tf, h, r, K, alpha, beta, q):   
    t_values, y_values_LVE = runge_kutta4_system(lotka_volterra_LVE, t0, y0, tf, h, r, r, K, alpha, beta, q)
    t_values, y_values = runge_kutta4_system(lotka_volterra, t0, y0, tf, h, r, r, alpha, beta, q)

    N_values_LVE = y_values_LVE[:, 0]
    P_values_LVE = y_values_LVE[:, 1]
    
    N_values = y_values[:, 0]
    P_values = y_values[:, 1]
    
    dN_values_LVE = dN_dt_LVE(N_values_LVE, P_values_LVE, r, alpha, K)
    dP_values_LVE = dP_dt(N_values_LVE, P_values_LVE, beta, q)
    
    dN_values = dN_dt(N_values, P_values, r, alpha)
    dP_values = dP_dt(N_values, P_values, beta, q)
    
    plt.figure()
    plt.plot(N_values_LVE, dN_values_LVE, label='presas (LVE)', color = 'peru')
    plt.plot(P_values_LVE, dP_values_LVE, label='depredadores (LVE)', color = 'darkred')    
    plt.plot(N_values, dN_values, label='presas (original)', color = 'seagreen')
    plt.plot(P_values, dP_values, label='depredadores (original)', color = 'dodgerblue')    
    
    plt.xlabel('Tamaño Poblacional', fontsize=14)
    plt.ylabel('Tamaño Poblacional', fontsize=14)
    plt.title('Evolución del tamaño poblacional', fontsize=16)
    plt.legend(handlelength=0.75)
    plt.grid(True)
    plt.show()
 
def lotka_q_r_K_constant(t, y, alpha, beta): 
    yp = (1 - alpha * y[1]) * y[0]
    yp = np.append(yp, (-1 + beta * y[0]) * y[1]) # q = 1
    return yp

def plano_de_fases(alpha, beta, y0_values, bool = True):
    N = np.linspace(0, 500, 100)
    P = np.linspace(0, 1000, 100)

    plt.figure(figsize=(8, 6))
    for y0 in y0_values:
        yrk = runge_kutta4_system(lotka_q_r_K_constant, 0, y0, 15, 0.01, alpha, beta )[1]
        plt.plot(yrk[:, 0], yrk[:, 1])
        
    isocline_p = 1 / beta
    isocline_n = 1 / alpha
    
    plt.axhline(y=isocline_n, linestyle = '--', color='limegreen')  # Asegura que isocline_n tenga la misma longitud que N
    plt.axvline(x=isocline_p, linestyle = '--', color='darkred')
    
    punto_eq = punto_equilibrio(1, alpha, beta, 1)
    plt.plot(punto_eq[0], punto_eq[1], 'o', color='teal', markersize=10, label='Punto de equilibrio')
    
    plt.title('Diagrama de Fases: comportamiento de ambas poblaciones en conjunto', fontsize=20)
    plt.xlabel('Población de Presas (N)', fontsize=18)
    plt.ylabel('Población de Predadores (P)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.show()
    
def plano_de_fases_LVE(y0, alpha, beta, r, q, K):

    N = np.linspace(0, K, 100)
    P = np.linspace(0, K, 100)
    
    
    yrk = runge_kutta4_system(lotka_volterra_LVE, 0, y0, 200, 0.01, r, r, K, alpha, beta, q)[1]
    plt.plot(yrk[:, 0], yrk[:, 1], color='lightsalmon')
        
    isocline_p = q / beta
    isocline_n = (r * (1 - N / K)) / alpha
    
    plt.plot(N, isocline_n, linestyle = '--', color='limegreen') 
    plt.axvline(x=isocline_p, linestyle = '--', color='darkred')
    
    plt.xlim(-2, 102)
    plt.ylim(7, 38)
     
    punto_eq = punto_equilibrio_LVE(r, K, alpha, beta, q)

    plt.plot(punto_eq[0], punto_eq[1], 'o', color='teal', markersize=5, label='Punto de equilibrio')
    
    plt.plot([], [], color='lightsalmon', label=f"α = {alpha}, β = {beta}, \nr = {r}, q = {q}, K = {K}")
     
    plt.title('Diagrama de Fases LVE', fontsize=20)
    plt.xlabel('Población de Presas (N)', fontsize=18)
    plt.ylabel('Población de Predadores (P)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=14, handlelength=0.75)
    if bool:
        plt.show()

def plano_de_fase (alpha, beta, y0):

    plt.figure(figsize=(8, 6))
    t_values, yrk = runge_kutta4_system(lotka_q_r_K_constant, 0, y0, 15, 0.01, alpha, beta )
    plt.plot(yrk[:, 0], yrk[:, 1])

    punto_eq = punto_equilibrio(1, alpha, beta, 1)

    plt.plot(punto_eq[0], punto_eq[1], 'o', color='teal', markersize=10, alpha=0.2)
    
    plt.plot (y0[0], y0[1], 'o', color='black', markersize=10, label='p0 = (120, 50)')
    plt.plot(135.5, 92, 'o', color='green', markersize=10, label='p1 = (135, 92)')
    plt.plot(50, 271.5, 'o', color='purple', markersize=10, label='p2 = (50, 271)')
    plt.plot(11.1, 100, 'o', color='red', markersize=10, label='p3 = (11, 100)')
    plt.plot(20, 35, 'o', color='orange', markersize=10, label='p4 = (20, 35)')
   
    plt.xlabel('Población de Presas (N)', fontsize=18)
    plt.ylabel('Población de Predadores (P)', fontsize=18)
    
    plt.title('Trayectoria de Fase', fontsize=22)
    plt.legend(fontsize=10)
    plt.show()
 
h = 0.1
tf = 300
t0 = 0
   
r = 1 # Tasa de crecimiento de las presas
alpha = 0.05 # éxito en la caza, afecta al crecimiento de las presas
beta = 0.01 # éxito en la caza, afecta al crecimiento de los depredadores
q = 0.1 # tasa de mortalidad de los depredadores
K = 1000 # Capacidad de carga del ambiente para las presas
N0 = 100
P0 = 10
y0 = [N0, P0]

# caso a:
# r>0 y q/b < K

# caso b:
# r>0 y q/b > K

# caso c:
# r<0 y q/b < K

# caso d:
# r<0 y q/b > K


cases = {
    'a': {'coef': 'r>0 y q/b < K', 'r': 1.5, 'alpha': 0.08, 'K': 100, 'beta': 0.02, 'q': 0.15, 'title': 'Caso a', 'legend_loc': 'upper center', 'tf': 150, 'lim': [[-5, 100], [0, 150]] , 'puntos_iniciales': [[26.1, 24.4], [78, 26], [95.4, 38.9], [71.3, 81.6]]},
    'b': {'coef': 'r>0 y q/b > K', 'r': 0.1, 'alpha': 0.005, 'K': 400, 'beta': 0.0009, 'q': 0.4, 'title': 'Caso b', 'legend_loc': 'center right', 'tf': 200, 'lim': [[-15, 150], [0, 900]], 'puntos_iniciales': [ [655, 128.4], [720, 58], [788, 91.8], [527, 3], [119, 143.2], [750, 20]]},
    'c': {'coef': 'r<0 y q/b < K', 'r': 0.2, 'alpha': 0.09, 'K': 1000, 'beta': 0.003, 'q': 0.005, 'title': 'Caso c', 'legend_loc': 'upper right', 'tf': 350, 'lim':[[-30, 50], [-20, 100]], 'puntos_iniciales': [[53.5, 27.7], [50.5, -18.9], [-15.6, 33.3], [-13.5, -20.3], [11.6, -7.9], [18.5, 14.6]]},
    'd': {'coef': 'r<0 y q/b > K', 'r': -2, 'alpha': 0.005, 'K': 5000, 'beta': 0.0005, 'q': 0.4, 'title': 'Caso d', 'legend_loc': 'lower left', 'tf': 50, 'lim': [[-500, 50], [-10, 900] ], 'puntos_iniciales': [[877, -257], [270, -119], [495, -462], [869, -473], [708, -356], [836, -348], [433, -99], [752, -342]] }
}


alpha_1 = 0.01
beta_1 = 0.02
y0_values = ([1, 1], [10, 10], [20, 20], [30, 30], [40, 40])
y0_values2 = ([2, 1], [20, 10], [40, 20], [60, 30], [80,40])

def main():
   
    graficar_soluciones_rk_varias(t0, N0, P0, h, cases)

    plano_de_fases(alpha_1, beta_1, y0_values ) 
    plano_de_fase(alpha_1, beta_1, [120, 50])
    plano_de_fases_LVE(y0,  cases['a']['alpha'], cases['a']['beta'], cases['a']['r'], cases['a']['q'], cases['a']['K'])
    
    isoclinas_y_campo_vectorial_varias(cases)
    isoclinas_y_campo_vectorial_LVE_varias(cases)
    
    tamaño_poblacional(t0, y0, tf, h, cases['a']['r'], cases['a']['K'], cases['a']['alpha'], cases['a']['beta'], cases['a']['q'])
    
    dos_variaciones(t0, y0, cases['a']['tf'], h, cases)
     
if __name__ == '__main__':
    main()

