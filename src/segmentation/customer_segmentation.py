from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class CustomerSegmentation:
  def __init__(self, n_clusters=5, random_state=42):
    """
    Inicializa el modelo de segmentación de clientes.

    Args:
        n_clusters (int): Número de segmentos a crear
        random_state (int): Semilla para reproducibilidad
    """
    self.n_clusters = n_clusters
    self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
    self.scaler = StandardScaler()
    self.feature_names = None
    self.is_fitted = False
    self.cluster_centers_ = None
    self.labels_ = None

  def fit_kmeans(self, data, feature_names=None):
    """
    Entrena el modelo de segmentación.

    Args:
        data (array-like): Datos para entrenamiento
        feature_names (list): Nombres de las características

    Returns:
        self: Para encadenamiento de métodos
    """
    # Guardar nombres de características
    if isinstance(data, pd.DataFrame):
      self.feature_names = data.columns.tolist()
      data_values = data.values
    else:
      self.feature_names = feature_names or [f"Feature_{i}" for i in
                                             range(data.shape[1])]
      data_values = data

    # Estandarizar datos
    self.data_scaled = self.scaler.fit_transform(data_values)

    # Ajustar modelo
    self.model.fit(self.data_scaled)

    # Guardar resultados
    self.cluster_centers_ = self.model.cluster_centers_
    self.labels_ = self.model.labels_
    self.is_fitted = True

    return self

  def predict(self, data):
    """
    Predice los segmentos para nuevos datos.

    Args:
        data (array-like): Datos para predecir

    Returns:
        array: Etiquetas de segmentos predichas
    """
    if not self.is_fitted:
      raise ValueError("El modelo debe ser entrenado primero usando fit_kmeans")

    # Preparar datos
    if isinstance(data, pd.DataFrame):
      data = data.values

    # Estandarizar y predecir
    data_scaled = self.scaler.transform(data)
    return self.model.predict(data_scaled)

  def get_cluster_profiles(self, data=None):
    """
    Obtiene los perfiles de cada segmento.

    Args:
        data (DataFrame): Datos originales (opcional)

    Returns:
        DataFrame: Perfiles de los segmentos
    """
    if not self.is_fitted:
      raise ValueError("El modelo debe ser entrenado primero usando fit_kmeans")

    # Convertir centros de clusters a espacio original
    centers_original = self.scaler.inverse_transform(self.cluster_centers_)

    # Crear DataFrame con perfiles
    profiles = pd.DataFrame(
        centers_original,
        columns=self.feature_names,
        index=[f"Segmento_{i + 1}" for i in range(self.n_clusters)]
    )

    if data is not None and isinstance(data, pd.DataFrame):
      # Agregar tamaño y porcentaje de cada segmento
      segment_sizes = pd.Series(self.labels_).value_counts().sort_index()
      profiles['Tamaño'] = segment_sizes.values
      profiles['Porcentaje'] = (
            segment_sizes.values / len(self.labels_) * 100).round(2)

    return profiles

  def evaluate_model(self, data=None):
    """
    Evalúa la calidad del clustering usando múltiples métricas.

    Args:
        data (array-like): Datos para evaluación (opcional)

    Returns:
        dict: Métricas de evaluación
    """
    if not self.is_fitted:
      raise ValueError("El modelo debe ser entrenado primero usando fit_kmeans")

    data_to_evaluate = self.data_scaled if data is None else self.scaler.transform(
      data)

    metrics = {
      'inercia': self.model.inertia_,
      'silhouette': silhouette_score(data_to_evaluate, self.labels_),
      'calinski_harabasz': calinski_harabasz_score(data_to_evaluate,
                                                   self.labels_)
    }

    return metrics

  def get_segment_characteristics(self, data):
    """
    Obtiene características distintivas de cada segmento.

    Args:
        data (DataFrame): Datos originales con características

    Returns:
        dict: Características principales por segmento
    """
    if not isinstance(data, pd.DataFrame):
      raise ValueError(
        "data debe ser un DataFrame con las características originales")

    characteristics = {}

    for cluster in range(self.n_clusters):
      # Obtener datos del segmento
      segment_data = data[self.labels_ == cluster]

      # Calcular estadísticas del segmento
      stats = segment_data.agg(['mean', 'std', 'min', 'max'])

      # Comparar con la media global
      global_means = data.mean()
      relative_importance = (stats.loc['mean'] - global_means) / global_means

      # Guardar características más distintivas
      top_features = relative_importance.abs().nlargest(3)

      characteristics[f"Segmento_{cluster + 1}"] = {
        'size': len(segment_data),
        'percentage': len(segment_data) / len(data) * 100,
        'top_features': top_features.to_dict(),
        'statistics': stats.to_dict()
      }

    return characteristics

  def get_optimal_clusters(self, data, max_clusters=10):
    """
    Determina el número óptimo de clusters usando el método del codo.

    Args:
        data (array-like): Datos para análisis
        max_clusters (int): Número máximo de clusters a probar

    Returns:
        dict: Inercia para diferentes números de clusters
    """
    if isinstance(data, pd.DataFrame):
      data = data.values

    data_scaled = self.scaler.fit_transform(data)
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(data_scaled)
      inertias.append(kmeans.inertia_)
      silhouette_scores.append(
          silhouette_score(data_scaled, kmeans.labels_)
      )

    return {
      'n_clusters': list(range(2, max_clusters + 1)),
      'inertia': inertias,
      'silhouette_score': silhouette_scores
    }