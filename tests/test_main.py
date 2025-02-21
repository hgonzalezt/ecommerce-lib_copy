import unittest
import pandas as pd
from src.MainFacade import MainFacade  # Asegúrate de que esta ruta es correcta

class TestMainFacade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Configurar instancia antes de las pruebas """
        cls.facade = MainFacade()

    def test_train_model_segmentation(self):
        """ Prueba entrenamiento de segmentación de clientes """
        data = pd.DataFrame({
            'edad': [25, 30, 35, 40, 45],
            'salario': [30000, 45000, 50000, 60000, 80000],
            'frecuencia_compra': [1, 2, 3, 4, 5]
        })
        data.to_csv("test_data.csv", index=False)
        result = self.facade.train_model("test_data.csv", "segmentation", {})
        self.assertIn("silhouette", result)  # Verificar métrica de evaluación

    def test_save_and_load_model(self):
        """ Prueba de guardado y carga de modelo """
        self.facade.model = self.facade.segmentation
        self.facade.save_model("test_model.pkl")
        self.facade.load_model("test_model.pkl")
        self.assertIsNotNone(self.facade.model)

    def test_test_model(self):
        """ Prueba de predicción con modelo cargado """
        data = pd.DataFrame({
            'edad': [29, 41],
            'salario': [32000, 70000],
            'frecuencia_compra': [2, 5]
        })
        data.to_csv("test_data_pred.csv", index=False)

        # Entrenar el modelo antes de predecir
        train_data = pd.DataFrame({
            'edad': [25, 30, 35, 40, 45],
            'salario': [30000, 45000, 50000, 60000, 80000],
            'frecuencia_compra': [1, 2, 3, 4, 5]
        })
        self.facade.segmentation.fit_kmeans(train_data)  # ✅ Ahora el modelo está entrenado

        self.facade.model = self.facade.segmentation  # Asignar el modelo entrenado
        predictions = self.facade.test_model("test_data_pred.csv")

        self.assertEqual(len(predictions), 2)  # Debe predecir para 2 entradas


if __name__ == "__main__":
    unittest.main()
