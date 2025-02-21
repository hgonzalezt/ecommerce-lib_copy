import pandas as pd
from src.preprocessor.data_preprocesor import DataPreprocessor
from src.analysis.market_basket import MarketBasketAnalysis
from src.dimension.pca_analysis import PCAAnalysis
from src.segmentation.customer_segmentation import CustomerSegmentation
import joblib
class MainFacade:
    """
    Clase principal que actúa como fachada para proporcionar una interfaz unificada.
    """

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.mba = MarketBasketAnalysis()
        self.pca = PCAAnalysis()
        self.segmentation = CustomerSegmentation(n_clusters=3)
        self.model = None

    def train_model(self, data_path, model_type, params):
        """
        Entrena un modelo seleccionado con los datos proporcionados.
        """
        # data = pd.read_csv(data_path, dtype=params.get('dtype', {}))
        data = pd.read_csv(data_path)  # Versión original

        if model_type == 'segmentation':
            self.segmentation.fit_kmeans(data)
            metrics = self.segmentation.evaluate_model()
        elif model_type == 'pca':
            self.pca.fit_pca(data)
            metrics = self.pca.get_explained_variance_ratio()
        elif model_type == 'market_basket':
            transactions = data.values.tolist()
            self.mba.fit_apriori(transactions, min_support=params.get('min_support', 0.25))
            metrics = self.mba.frequent_itemsets
        else:
            raise ValueError("Modelo no soportado")

        return metrics

    def save_model(self, file_path):
        """
        Guarda el modelo entrenado en un archivo.
        """
        # joblib.dump(self.model, file_path, compress=3)
        joblib.dump(self.model, file_path)  # Versión original

    def load_model(self, file_path):
        """
        Carga un modelo previamente guardado desde un archivo.
        """
        # self.model = joblib.load(file_path, mmap_mode='r')
        self.model = joblib.load(file_path)  # Versión original

    def test_model(self, test_data_path):
        """
        Prueba el modelo cargado con nuevos datos de prueba.
        """
        # test_data = pd.read_csv(test_data_path, dtype={'edad': 'int8', 'salario': 'int32'})
        test_data = pd.read_csv(test_data_path)  # Versión original

        if not self.model:
            raise ValueError("El modelo no ha sido cargado o entrenado")

        predictions = self.model.predict(test_data)
        return predictions
