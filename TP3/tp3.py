import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Función para calcular la matriz de similaridad
def calcular_matriz_similaridad(X, sigma):
    """
    Calcula la matriz de similaridad para el dataset X utilizando
    una función gaussiana basada en la distancia euclidiana.
    
    Args:
        X (array-like): Matriz de datos.
        sigma (float): Parámetro de la función gaussiana.
    
    Returns:
        K (array-like): Matriz de similaridad.
    """
    distancias = squareform(pdist(X, 'euclidean'))  # Calcula distancias euclidianas par-a-par
    K = np.exp(-distancias**2 / (2 * sigma**2))  # Aplica la función gaussiana a las distancias
    return K

# Función para aplicar PCA y reducir la dimensionalidad
def reducir_dimensionalidad(X, d):
    """
    Aplica PCA para reducir la dimensionalidad del dataset X a d dimensiones.
    
    Args:
        X (array-like): Matriz de datos.
        d (int): Número de dimensiones deseadas.
    
    Returns:
        Z (array-like): Matriz de datos reducida.
        pca (PCA object): Objeto PCA ajustado.
    """
    pca = PCA(n_components=d)  # Crear el objeto PCA con d componentes
    Z = pca.fit_transform(X)  # Ajustar el PCA a los datos y transformarlos
    return Z, pca

def variar_d_y_graficar(X, d, sigma=1.0):
    similaridades = {}
    for dimension in d:
        Z, pca = reducir_dimensionalidad(X, dimension)  # Reducir dimensionalidad a d = dimension
        similaridades[dimension] = calcular_matriz_similaridad(Z, sigma)  # Calcular matriz de similaridad en espacio reducido
        print(f"Para d = {dimension}, la varianza explicada es: {np.sum(pca.explained_variance_ratio_)}")

    K_original = calcular_matriz_similaridad(X, sigma)

    fig, axes = plt.subplots(2, len(d), figsize=(20, 10))  # Crear una figura con subplots
    axes = axes.ravel()  # Aplanar los ejes para fácil acceso

    for i, dimension in enumerate(d):
        sns.heatmap(K_original, ax=axes[i], cmap='viridis', cbar=True)
        axes[i].set_title(f'Original (d={dimension})')
        axes[i].set_xlabel('Muestras')
        axes[i].set_ylabel('Muestras')

        sns.heatmap(similaridades[dimension], ax=axes[len(d) + i], cmap='viridis', cbar=True)
        axes[len(d) + i].set_title(f'Reducido a {dimension} dim. (d={dimension})')
        axes[len(d) + i].set_xlabel('Muestras')
        axes[len(d) + i].set_ylabel('Muestras')

    plt.tight_layout()  # Ajustar el layout para evitar solapamiento
    plt.show()  # Mostrar la figura

def variar_sigma_y_graficar(d, sigmas, X):
    similaridades = {}
    for sigma in sigmas:
        Z, pca = reducir_dimensionalidad(X, d)  # Reducir dimensionalidad a d = dimension
        similaridades[sigma] = calcular_matriz_similaridad(Z, sigma)  # Calcular matriz de similaridad en espacio reducido
        print(f"Para sigma = {sigma}, la varianza explicada es: {np.sum(pca.explained_variance_ratio_)}")

    K_original = {sigma: calcular_matriz_similaridad(X, sigma) for sigma in sigmas}

    fig, axes = plt.subplots(2, len(sigmas), figsize=(20, 10))  # Crear una figura con subplots
    axes = axes.ravel()  # Aplanar los ejes para fácil acceso

    for i, sigma in enumerate(sigmas):
        sns.heatmap(K_original[sigma], ax=axes[i], cmap='viridis', cbar=True)
        axes[i].set_title(f'Original (sigma={sigma})')
        axes[i].set_xlabel('Muestras')
        axes[i].set_ylabel('Muestras')

        sns.heatmap(similaridades[sigma], ax=axes[len(sigmas) + i], cmap='viridis', cbar=True)
        axes[len(sigmas) + i].set_title(f'Reducido a {d} dim. (sigma={sigma})')
        axes[len(sigmas) + i].set_xlabel('Muestras')
        axes[len(sigmas) + i].set_ylabel('Muestras')

    plt.tight_layout()  # Ajustar el layout para evitar solapamiento
    plt.show()  # Mostrar la figura

def sigma_como_varianza_graficar(X, d):
    similaridades = {}
    varianzas = {}
    for dimension in d:
        Z, pca = reducir_dimensionalidad(X, dimension)
        varianza = np.sum(pca.explained_variance_ratio_)
        varianzas[dimension] = varianza
        similaridades[dimension] = calcular_matriz_similaridad(Z, varianza)
        print(f"Para d = {dimension}, sigma = varianza explicada = {varianza}")

    K_original = {dimension: calcular_matriz_similaridad(X, varianza) for dimension, varianza in varianzas.items()}

    fig, axes = plt.subplots(2, len(d), figsize=(20, 10))  # Crear una figura con subplots
    axes = axes.ravel()  # Aplanar los ejes para fácil acceso

    for i, dimension in enumerate(d):
        sns.heatmap(K_original[dimension], ax=axes[i], cmap='viridis', cbar=True)
        axes[i].set_title(f'Original (d={dimension})')
        axes[i].set_xlabel('Muestras')
        axes[i].set_ylabel('Muestras')

        sns.heatmap(similaridades[dimension], ax=axes[len(d) + i], cmap='viridis', cbar=True)
        axes[len(d) + i].set_title(f'Reducido a {dimension} dim. (d={dimension}, sigma=varianza)')
        axes[len(d) + i].set_xlabel('Muestras')
        axes[len(d) + i].set_ylabel('Muestras')

    plt.tight_layout()  # Ajustar el layout para evitar solapamiento
    plt.show()  # Mostrar la figura

file_path = 'dataset.csv'  # Ruta del archivo CSV
X = pd.read_csv(file_path).values  # Leer el archivo CSV en un DataFrame y convertirlo a una matriz NumPy

# Ver las dimensiones del dataset
n, p = X.shape
print(f"Dimensiones del dataset: {n} muestras y {p} características")

d = [2, 6, 10]  # Dimensión reducida fija
sigmas = [0.1, 10, 100, 10000]  # Lista de valores de sigma a evaluar

def main():
    for dimension in d:
        variar_sigma_y_graficar(dimension, sigmas, X)
    
    variar_d_y_graficar(X, d, sigma=1.0)
    sigma_como_varianza_graficar(X, d)

if __name__ == "__main__":
    main()
