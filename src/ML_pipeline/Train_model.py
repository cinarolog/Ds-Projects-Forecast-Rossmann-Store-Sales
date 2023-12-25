import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib
from ML_pipeline.param_grid import choose_param_grid

def train_model(x_train, x_test, y_train, y_test, model_name, path):
    """
    Verilen eğitim setiyle model eğitir, test seti üzerinde tahmin yapar ve eğitilmiş modeli bir dosyaya kaydeder.

    Parametreler:
    - x_train: Eğitim veri seti özellikleri.
    - x_test: Test veri seti özellikleri.
    - y_train: Eğitim veri seti hedef değişkeni.
    - y_test: Test veri seti hedef değişkeni.
    - model_name: Kullanılacak modelin adı. ('linear_reg', 'SGD_reg', 'RF_reg', 'dtree_reg', 'ridge', 'lasso', 'gb_reg', 'knn_reg', 'svm_reg', 'elastic_net', 'huber_reg')
    - path: Eğitilmiş modelin kaydedileceği dosya yolu.

    Dönüş:
    - pred: Test seti üzerinde yapılan tahminler.
    """

    model_dict = {
        'linear_reg': LinearRegression,
        'SGD_reg': SGDRegressor,
        'RF_reg': RandomForestRegressor,
        'dtree_reg': DecisionTreeRegressor,
        'ridge': Ridge,
        'lasso': Lasso,
        'gb_reg': GradientBoostingRegressor,
        'knn_reg': KNeighborsRegressor,
        'svm_reg': SVR,
        'elastic_net': ElasticNet,
        'huber_reg': HuberRegressor,
    }

    # Geçersiz model adı verildiyse hata yükselt
    if model_name not in model_dict:
        raise ValueError(f"Only these options for model_name are allowed: {list(model_dict.keys())}")

    # Seçilen modeli oluştur
    model = model_dict[model_name]()

    # Modeli eğit
    model.fit(x_train, y_train)

    # Test seti üzerinde tahmin yap
    pred = model.predict(x_test)

    # Eğitilmiş modeli dosyaya kaydet
    joblib.dump(model, path)
    print(f'Model saved as a pickle file in {path}')

    # Test seti üzerinde yapılan tahminleri döndür
    return pred
