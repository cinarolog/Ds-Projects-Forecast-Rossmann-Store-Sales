import pandas as pd
import numpy as np

def feature_importance(X_train_cols, model):
    """
    Verilen modelin özellik önem değerlerini kullanarak bir özellik önem sıralaması oluşturur.

    Parametreler:
    - X_train_cols: Eğitim veri setindeki özellik sütunlarının isimlerini içeren liste.
    - model: Eğitilmiş makine öğrenimi modeli.

    Dönüş:
    - feature_importance_df: Özellik önem sıralamasını içeren DataFrame.
    """

    # Modelin özellik önem değerlerini al
    feature_importance = model.feature_importances_

    # Her bir özellik için özellik önem değerlerini yuvarla
    feature_importance_value = [round(value, 5) for value in feature_importance]

    # Özellik adları ve önem değerlerini içeren bir DataFrame oluştur
    feature_importance_df = pd.DataFrame({"Features": X_train_cols, "Values": feature_importance_value})

    # Özellik önem değerlerine göre DataFrame'i sırala (büyükten küçüğe)
    feature_importance_df.sort_values(by=["Values"], inplace=True, ascending=False)

    # Oluşturulan özellik önem sıralamasını döndür
    return feature_importance_df
