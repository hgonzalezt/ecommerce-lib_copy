from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class PCAAnalysis:
  def __init__(self, n_components=None):
    """
    Inicializa el análisis PCA.

    Args:
        n_components (int): Número de componentes principales a mantener.
                          Si es None, se mantienen todos los componentes.
    """
    self.pca = PCA(n_components=n_components)
    self.scaler = StandardScaler()
    self.feature_names = None
    self.is_fitted = False

  def fit_pca(self, data, feature_names=None):
    """
    Ajusta el modelo PCA a los datos.

    Args:
        data (array-like): Datos para ajustar el PCA
        feature_names (list): Nombres de las características (opcional)

    Returns:
        self: Permite encadenar métodos
    """
    # Guardar nombres de características
    if isinstance(data, pd.DataFrame):
      self.feature_names = data.columns.tolist()
      data = data.values
    else:
      self.feature_names = feature_names or [f"Feature_{i}" for i in
                                             range(data.shape[1])]

    # Estandarizar los datos
    data_scaled = self.scaler.fit_transform(data)

    # Ajustar PCA
    self.pca.fit(data_scaled)
    self.is_fitted = True

    return self

  def transform(self, data):
    """
    Transforma los datos usando el PCA ajustado.

    Args:
        data (array-like): Datos a transformar

    Returns:
        array: Datos transformados
    """
    if not self.is_fitted:
      raise ValueError("El modelo PCA debe ser ajustado primero usando fit_pca")

    # Convertir a numpy si es DataFrame
    if isinstance(data, pd.DataFrame):
      data = data.values

    # Estandarizar y transformar
    data_scaled = self.scaler.transform(data)
    return self.pca.transform(data_scaled)

  def fit_transform(self, data, feature_names=None):
    """
    Ajusta el modelo y transforma los datos en un solo paso.

    Args:
        data (array-like): Datos para ajustar y transformar
        feature_names (list): Nombres de las características (opcional)

    Returns:
        array: Datos transformados
    """
    return self.fit_pca(data, feature_names).transform(data)

  def get_explained_variance_ratio(self):
    """
    Obtiene el ratio de varianza explicada por cada componente.

    Returns:
        array: Ratio de varianza explicada
    """
    if not self.is_fitted:
      raise ValueError("El modelo PCA debe ser ajustado primero usando fit_pca")

    return self.pca.explained_variance_ratio_

  def get_cumulative_variance_ratio(self):
    """
    Obtiene el ratio de varianza explicada acumulada.

    Returns:
        array: Ratio de varianza explicada acumulada
    """
    return np.cumsum(self.get_explained_variance_ratio())

  def get_feature_importance(self):
    """
    Obtiene la importancia de cada característica original en los componentes.

    Returns:
        DataFrame: Importancia de características por componente
    """
    if not self.is_fitted:
      raise ValueError("El modelo PCA debe ser ajustado primero usando fit_pca")

    # Obtener los loadings (coeficientes de las características originales)
    loadings = pd.DataFrame(
        self.pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(self.pca.n_components_)],
        index=self.feature_names
    )

    return loadings

  def get_optimal_components(self, variance_threshold=0.95):
    """
    Determina el número óptimo de componentes para explicar un porcentaje de varianza.

    Args:
        variance_threshold (float): Porcentaje de varianza a explicar (0-1)

    Returns:
        int: Número de componentes necesarios
    """
    cumsum = self.get_cumulative_variance_ratio()
    return np.argmax(cumsum >= variance_threshold) + 1

  def inverse_transform(self, transformed_data):
    """
    Revierte la transformación PCA para obtener los datos originales aproximados.

    Args:
        transformed_data (array-like): Datos en el espacio PCA

    Returns:
        array: Datos en el espacio original
    """
    if not self.is_fitted:
      raise ValueError("El modelo PCA debe ser ajustado primero usando fit_pca")

    # Invertir PCA y estandarización
    data_scaled = self.pca.inverse_transform(transformed_data)
    return self.scaler.inverse_transform(data_scaled)

  def plot_explained_variance(self):
    """
    Genera un gráfico de varianza explicada por componente.

    Returns:
        dict: Datos del gráfico para visualización
    """
    explained_variance = self.get_explained_variance_ratio()
    cumulative_variance = self.get_cumulative_variance_ratio()

    plot_data = {
      'components': list(range(1, len(explained_variance) + 1)),
      'explained_variance': explained_variance * 100,
      'cumulative_variance': cumulative_variance * 100
    }

    return plot_data

  def get_component_coordinates(self, data):
    """
    Obtiene las coordenadas de los datos en el espacio PCA.

    Args:
        data (array-like): Datos a transformar

    Returns:
        DataFrame: Coordenadas en el espacio PCA
    """
    transformed_data = self.transform(data)
    return pd.DataFrame(
        transformed_data,
        columns=[f'PC{i + 1}' for i in range(transformed_data.shape[1])]
    )