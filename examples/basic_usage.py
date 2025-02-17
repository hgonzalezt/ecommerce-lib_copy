import pandas as pd
from src.preprocessor.data_preprocesor import DataPreprocessor
from src.analysis.market_basket import MarketBasketAnalysis
from src.dimension.pca_analysis import PCAAnalysis

print("=== PRUEBA DEL PREPROCESSOR ===")

# Crear datos de ejemplo
'''data = {
    'edad': [25, 30, 35, 40, 45],
    'salario': [30000, 45000, 50000, 60000, 80000],
    'categoria': ['A', 'B', 'A', 'C', 'B'],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona']
}'''

# Cargar los archivos CSV
data = pd.read_csv('../data/data.csv')
data_pca = pd.read_csv('../data/data_pca.csv')
data_segmentation = pd.read_csv('../data/data_segmentation.csv')
df_transactions1 = pd.read_csv('../data/transactions.csv', header=None)
transactions = df_transactions1.values.tolist()  # Asumiendo que df_transactions es un DataFrame


df = pd.DataFrame(data)

# Crear instancia del preprocessor
preprocessor = DataPreprocessor()

# Probar la estandarización
df_standardized = preprocessor.standardize_data(df, columns=['edad', 'salario'])
print("\nDatos estandarizados:")
print(df_standardized)

# Probar la normalización
df_normalized = preprocessor.normalize_data(df, columns=['edad', 'salario'])
print("\nDatos normalizados:")
print(df_normalized)

# Probar la codificación de categóricas
df_encoded = preprocessor.encode_categorical(df, columns=['categoria', 'ciudad'])
print("\nDatos codificados:")
print(df_encoded)


print("\n=== PRUEBA DEL MARKET BASKET ANALYSIS ===")

# Crear datos de ejemplo (transacciones de supermercado)
'''
transactions = [
    ['pan', 'leche', 'huevos'],
    ['pan', 'café', 'azúcar'],
    ['leche', 'café', 'galletas'],
    ['pan', 'leche', 'café'],
    ['pan', 'huevos', 'galletas'],
    ['leche', 'café', 'azúcar'],
    ['pan', 'leche', 'café', 'huevos'],
    ['pan', 'café', 'azúcar', 'galletas']
]
'''
# Crear instancia del análisis
mba = MarketBasketAnalysis()

# Encontrar conjuntos frecuentes
mba.fit_apriori(transactions, min_support=0.25)
print("\nConjuntos frecuentes encontrados:")
print(mba.frequent_itemsets)

# Generar reglas de asociación
rules = mba.generate_rules(min_confidence=0.5, min_lift=1.0)
print("\nReglas de asociación generadas:")
print(rules)

# Obtener las mejores reglas por lift
print("\nTop 5 reglas por lift:")
print(mba.get_top_rules(5))

# Hacer predicciones
basket = ['pan', 'leche']
recommendations = mba.predict_next_items(basket, n_recommendations=3)
print(f"\nRecomendaciones para la canasta {basket}:")
for item, score in recommendations.items():
    print(f"- {item}: {score:.2f}")


print("\n=== PRUEBA DEL PCA ANALYSIS ===")

# Crear datos de ejemplo más complejos para PCA
'''
data_pca = {
    'altura': [170, 175, 160, 180, 165, 172, 168, 185],
    'peso': [70, 75, 55, 85, 60, 72, 65, 88],
    'edad': [25, 28, 22, 30, 24, 27, 26, 32],
    'ingreso': [30000, 45000, 25000, 55000, 28000, 42000, 35000, 60000],
    'gasto_mensual': [1000, 1500, 800, 2000, 900, 1400, 1200, 2200]
}
'''
df_pca = pd.DataFrame(data_pca)

# Crear y ajustar PCA
pca = PCAAnalysis(n_components=3)
pca.fit_pca(df_pca)

# Obtener y mostrar la varianza explicada
explained_variance = pca.get_explained_variance_ratio()
print("\nVarianza explicada por componente:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

# Mostrar importancia de características
feature_importance = pca.get_feature_importance()
print("\nImportancia de características por componente:")
print(feature_importance)

# Transformar los datos
transformed_data = pca.transform(df_pca)
print("\nDatos transformados (primeros 3 componentes):")
print(pd.DataFrame(transformed_data[:, :3], columns=['PC1', 'PC2', 'PC3']).round(3))

# Obtener número óptimo de componentes para 95% de varianza
n_components = pca.get_optimal_components(0.95)
print(f"\nNúmero de componentes necesarios para explicar 95% de la varianza: {n_components}")

# Obtener datos del gráfico de varianza explicada
plot_data = pca.plot_explained_variance()
print("\nDatos para el gráfico de varianza explicada:")
print("Varianza explicada por componente (%):", plot_data['explained_variance'].round(1))
print("Varianza explicada acumulada (%):", plot_data['cumulative_variance'].round(1))


print("\n=== PRUEBA DE CUSTOMER SEGMENTATION ===")

# Importar CustomerSegmentation
from src.segmentation.customer_segmentation import CustomerSegmentation

# Crear datos de ejemplo más realistas para segmentación
'''
data_segmentation = {
    'edad': [25, 35, 45, 28, 55, 32, 42, 38, 48, 30, 60, 35, 42, 28, 45],
    'ingreso_anual': [30000, 45000, 80000, 35000, 90000, 42000, 65000, 55000, 75000, 38000, 95000, 48000, 62000, 32000, 70000],
    'gasto_mensual': [800, 1200, 2500, 900, 3000, 1100, 1800, 1500, 2200, 1000, 3500, 1300, 1700, 850, 2000],
    'frecuencia_compra': [2, 3, 5, 2, 6, 3, 4, 4, 5, 2, 6, 3, 4, 2, 4],
    'antiguedad_años': [1, 3, 8, 2, 10, 4, 6, 5, 7, 2, 12, 4, 5, 1, 6]
}
'''
df_segmentation = pd.DataFrame(data_segmentation)

# Crear y ajustar modelo de segmentación
segmentation = CustomerSegmentation(n_clusters=3)
segmentation.fit_kmeans(df_segmentation)

# Obtener perfiles de segmentos
profiles = segmentation.get_cluster_profiles(df_segmentation)
print("\nPerfiles de segmentos:")
print(profiles)

# Evaluar modelo
metrics = segmentation.evaluate_model()
print("\nMétricas de evaluación:")
print(f"Inercia: {metrics['inercia']:.2f}")
print(f"Silhouette Score: {metrics['silhouette']:.3f}")
print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")

# Obtener características de segmentos
characteristics = segmentation.get_segment_characteristics(df_segmentation)
print("\nCaracterísticas principales por segmento:")
for segment, info in characteristics.items():
    print(f"\n{segment}:")
    print(f"Tamaño: {info['size']} clientes ({info['percentage']:.1f}%)")
    print("Características distintivas:")
    for feature, value in info['top_features'].items():
        print(f"- {feature}: {value:+.2f}")

# Encontrar número óptimo de clusters
optimal_clusters = segmentation.get_optimal_clusters(df_segmentation, max_clusters=8)
print("\nAnálisis de número óptimo de clusters:")
print("Número de clusters -> Silhouette Score:")
for n, score in zip(optimal_clusters['n_clusters'], optimal_clusters['silhouette_score']):
    print(f"{n} clusters: {score:.3f}")