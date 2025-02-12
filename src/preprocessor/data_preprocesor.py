import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class DataPreprocessor:
  def __init__(self):
    """
    Inicializa los objetos necesarios para el preprocesamiento
    """
    self.standard_scaler = StandardScaler()
    self.min_max_scaler = MinMaxScaler()
    self.label_encoders = {}  # Un diccionario para mantener encoders por columna

  def standardize_data(self, df, columns=None):
    """
    Estandariza las columnas numéricas del DataFrame (media 0, desviación estándar 1)

    Args:
        df (pd.DataFrame): DataFrame a estandarizar
        columns (list): Lista de columnas a estandarizar. Si es None, se procesan todas las numéricas

    Returns:
        pd.DataFrame: DataFrame con las columnas estandarizadas
    """
    df_copy = df.copy()

    # Si no se especifican columnas, usar todas las numéricas
    if columns is None:
      columns = df_copy.select_dtypes(include=['int64', 'float64']).columns

    # Verificar que haya columnas para estandarizar
    if len(columns) == 0:
      return df_copy

    # Estandarizar las columnas seleccionadas
    df_copy[columns] = self.standard_scaler.fit_transform(df_copy[columns])

    return df_copy

  def normalize_data(self, df, columns=None):
    """
    Normaliza las columnas numéricas del DataFrame al rango [0,1]

    Args:
        df (pd.DataFrame): DataFrame a normalizar
        columns (list): Lista de columnas a normalizar. Si es None, se procesan todas las numéricas

    Returns:
        pd.DataFrame: DataFrame con las columnas normalizadas
    """
    df_copy = df.copy()

    # Si no se especifican columnas, usar todas las numéricas
    if columns is None:
      columns = df_copy.select_dtypes(include=['int64', 'float64']).columns

    # Verificar que haya columnas para normalizar
    if len(columns) == 0:
      return df_copy

    # Normalizar las columnas seleccionadas
    df_copy[columns] = self.min_max_scaler.fit_transform(df_copy[columns])

    return df_copy

  def encode_categorical(self, df, columns=None):
    """
    Codifica variables categóricas usando LabelEncoder

    Args:
        df (pd.DataFrame): DataFrame con las columnas a codificar
        columns (list): Lista de columnas a codificar. Si es None, se procesan todas las categóricas

    Returns:
        pd.DataFrame: DataFrame con las columnas codificadas
    """
    df_copy = df.copy()

    # Si no se especifican columnas, usar todas las categóricas
    if columns is None:
      columns = df_copy.select_dtypes(include=['object']).columns

    # Codificar cada columna categórica
    for column in columns:
      if column not in self.label_encoders:
        self.label_encoders[column] = LabelEncoder()

      # Manejar valores nulos si existen
      mask = df_copy[column].notnull()
      df_copy.loc[mask, column] = self.label_encoders[column].fit_transform(
          df_copy.loc[mask, column]
      )

    return df_copy