from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

class MarketBasketAnalysis:
  def __init__(self):
    """
    Inicializa la clase MarketBasketAnalysis.
    """
    self.frequent_itemsets = None
    self.rules = None
    self._support_dict = {}

  def _prepare_data(self, data):
    """
    Prepara los datos para el análisis.

    Args:
        data: Puede ser:
            - Lista de listas con transacciones
            - DataFrame binario (0/1)
            - Series con listas de items

    Returns:
        pd.DataFrame: DataFrame en formato binario
    """
    # Si es una lista de transacciones
    if isinstance(data, list):
      # Obtener todos los items únicos
      unique_items = sorted(
        list(set(item for transaction in data for item in transaction)))

      # Crear DataFrame binario
      transactions_dict = {
        item: [1 if item in transaction else 0 for transaction in data]
        for item in unique_items
      }
      return pd.DataFrame(transactions_dict)

    # Si los datos ya son un DataFrame
    if isinstance(data, pd.DataFrame):
      if set(data.values.flatten()) <= {0, 1, True, False}:
        return data

    # Si es una Series con listas
    if isinstance(data, pd.Series):
      return pd.get_dummies(data.apply(pd.Series).stack()).groupby(
        level=0).max()

    raise ValueError("Formato de datos no soportado")

  def fit_apriori(self, data, min_support=0.01, use_colnames=True):
    """
    Aplica el algoritmo Apriori para encontrar conjuntos de items frecuentes.

    Args:
        data: Los datos de transacciones
        min_support (float): Soporte mínimo (0-1)
        use_colnames (bool): Si se usan los nombres de columnas

    Returns:
        self: Permite encadenar métodos
    """
    # Preparar los datos
    prepared_data = self._prepare_data(data)

    # Aplicar Apriori
    self.frequent_itemsets = apriori(
        prepared_data,
        min_support=min_support,
        use_colnames=use_colnames
    )

    # Convertir los frozensets a listas para mejor visualización
    self.frequent_itemsets['itemsets'] = self.frequent_itemsets[
      'itemsets'].apply(list)

    # Guardar soporte de items individuales
    self._support_dict = {
      item: support for item, support in
      zip(prepared_data.columns,
          prepared_data.mean().values)
    }

    return self

  def generate_rules(self, min_confidence=0.5, min_lift=1.0):
    """
    Genera reglas de asociación a partir de los conjuntos frecuentes.

    Args:
        min_confidence (float): Confianza mínima (0-1)
        min_lift (float): Lift mínimo (>0)

    Returns:
        pd.DataFrame: DataFrame con las reglas de asociación
    """
    if self.frequent_itemsets is None:
      raise ValueError("Debes ejecutar fit_apriori primero")

    # Convertir listas back a frozensets para association_rules
    frequent_itemsets_copy = self.frequent_itemsets.copy()
    frequent_itemsets_copy['itemsets'] = frequent_itemsets_copy[
      'itemsets'].apply(frozenset)

    self.rules = association_rules(
        frequent_itemsets_copy,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Filtrar por lift
    self.rules = self.rules[self.rules['lift'] >= min_lift]

    # Convertir frozensets a listas para mejor visualización
    self.rules['antecedents'] = self.rules['antecedents'].apply(list)
    self.rules['consequents'] = self.rules['consequents'].apply(list)

    # Ordenar por lift y confidence
    self.rules = self.rules.sort_values(['lift', 'confidence'],
                                        ascending=[False, False])

    return self.rules

  def get_top_rules(self, n=10, metric='lift'):
    """
    Obtiene las n mejores reglas según la métrica especificada.

    Args:
        n (int): Número de reglas a retornar
        metric (str): Métrica para ordenar ('support', 'confidence', 'lift')

    Returns:
        pd.DataFrame: Las n mejores reglas
    """
    if self.rules is None:
      raise ValueError("Debes generar las reglas primero con generate_rules()")

    return self.rules.nlargest(n, metric)

  def predict_next_items(self, items, n_recommendations=5):
    """
    Predice los siguientes items más probables dado un conjunto de items.

    Args:
        items (list): Lista de items en la canasta actual
        n_recommendations (int): Número de recomendaciones a retornar

    Returns:
        dict: Diccionario con items recomendados y sus scores
    """
    if self.rules is None:
      raise ValueError("Debes generar las reglas primero con generate_rules()")

    # Convertir items a set para búsqueda más eficiente
    items_set = set(items)

    # Filtrar reglas relevantes
    relevant_rules = self.rules[
      self.rules['antecedents'].apply(lambda x: set(x).issubset(items_set))
    ]

    if relevant_rules.empty:
      return {}

    # Calcular scores para cada consecuente
    recommendations = {}
    for _, rule in relevant_rules.iterrows():
      for item in rule['consequents']:
        if item not in items_set:
          if item not in recommendations:
            recommendations[item] = rule['lift'] * rule['confidence']
          else:
            recommendations[item] = max(
                recommendations[item],
                rule['lift'] * rule['confidence']
            )

    # Ordenar y retornar los top n
    return dict(
        sorted(recommendations.items(),
               key=lambda x: x[1],
               reverse=True)[:n_recommendations]
    )