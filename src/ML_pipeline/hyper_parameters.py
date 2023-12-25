from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from ML_pipeline.param_grid import choose_param_grid

def hyperparameter_optimization(X, y, model_dict, scoring='r2_score', cv=3):
    """
    Regresyon modelleri için hyperparameter optimizasyonu gerçekleştirir.

    Parametreler:
    - X: Özelliklerin bulunduğu DataFrame veya array.
    - y: Hedef değişkenin bulunduğu Series veya array.
    - model_dict: Model adları ve sınıflarını içeren sözlük.
    - param_grid: Hyperparameter aralıklarını içeren sözlük.
    - scoring: Optimizasyon metriği. Varsayılan 'neg_mean_squared_error'.
    - cv: Çapraz doğrulama kat sayısı. Varsayılan 3.

    Dönüş:
    - best_models: En iyi modelleri içeren sözlük.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_models = {}

    for model_name, model_class in model_dict.items():
        model = model_class()
        grid_search = GridSearchCV(model, choose_param_grid(model_name), scoring=scoring, cv=cv)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_

    return best_models


