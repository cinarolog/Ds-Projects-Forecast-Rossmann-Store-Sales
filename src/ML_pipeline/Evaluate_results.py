import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

def evaluate_results(y_test, y_pred):
    """
    Gerçek ve tahmin edilen değerlere dayalı olarak çeşitli regresyon performans metriklerini değerlendirir.

    Parametreler:
    - y_test: Gerçek değerlerin bulunduğu dizi.
    - y_pred: Tahmin edilen değerlerin bulunduğu dizi.

    Dönüş:
    - metrics: Bir sözlük, içinde 'r2_score', 'mae', 'rmse', 'mse' ve 'mape' performans metriklerini içerir.
    """

    # R^2 skoru hesapla
    r2 = r2_score(y_test, y_pred)

    # Ortalama Mutlak Hata (MAE) hesapla
    mae = mean_absolute_error(y_test, y_pred)

    # Karekök Ortalama Kare Hata (RMSE) hesapla
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    # Ortalama Kare Hata (MSE) hesapla
    mse = mean_squared_error(y_test, y_pred)

    # Ortalama Mutlak Yüzde Hata (MAPE) hesapla
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Hesaplanan metrikleri sözlükte sakla
    metrics = {'r2_score': r2,
               'mae': mae,
               'rmse': rmse,
               'mse': mse,
               'mape': mape}
    

    # Sonuçları döndür
    return metrics
