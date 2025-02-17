from setuptools import setup, find_packages

setup(
    name="ecommerce_recommender",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.6.0",
        "mlxtend>=0.23.0",
        "joblib>=1.4.2",
    ],
    author="David D.",
    author_email="david@email.com",
    description="Librería de recomendaciones para e-commerce",
    keywords="ecommerce, recommendations, machine learning",
)