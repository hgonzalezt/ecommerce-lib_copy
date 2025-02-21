import joblib
import json

# Cargar el modelo desde el archivo .pkl
modelo = joblib.load("test_model.pkl")

# Mostrar el tipo de objeto cargado
print(type(modelo))

# Si es un modelo de machine learning, puedes ver sus atributos
if hasattr(modelo, 'predict'):
    print("El modelo está listo para hacer predicciones.")

with open("modelo_info.txt", "w") as file:
    file.write(str(modelo))

# Convertir a JSON si es un diccionario
if isinstance(modelo, dict):
    with open("modelo.json", "w") as file:
        json.dump(modelo, file, indent=4)
    print("Archivo convertido a JSON: modelo.json")

# Cargar el archivo .pkl
pkl_file = "test_model.pkl"  # Asegúrate de usar la ruta correcta
json_file = "test_model.json"

# Cargar el contenido del archivo .pkl
modelo = joblib.load(pkl_file)

# Convertir el objeto a JSON si es serializable
try:
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(modelo, file, indent=4, ensure_ascii=False)
    print(f"✅ Archivo convertido a JSON correctamente: {json_file}")
except TypeError:
    print("❌ El objeto no se puede convertir directamente a JSON. Intentando convertir atributos clave...")

    # Si el objeto no es serializable, intentamos convertir solo los atributos básicos
    modelo_dict = {attr: str(getattr(modelo, attr)) for attr in dir(modelo) if not attr.startswith("_")}

    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(modelo_dict, file, indent=4, ensure_ascii=False)

    print(f"✅ Se guardaron los atributos del modelo en: {json_file}")