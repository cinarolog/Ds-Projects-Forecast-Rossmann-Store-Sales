import pandas as pd
import numpy as np

def impute(df, col, method, value=0):
    """
    Verilen DataFrame içinde belirli bir sütundaki eksik değerleri doldurur.

    Parametreler:
    - df: Pandas DataFrame, eksik değerleri içeren veri seti.
    - col: str, eksik değerleri doldurulacak sütun adı.
    - method: str, eksik değerleri doldurmak için kullanılacak yöntem ('mean', 'median', 'mode', 'value').
    - value: Değer, 'value' yöntemi seçildiğinde kullanılacak özel bir değer.

    Dönüş:
    - df: Eksik değerleri doldurulmuş DataFrame.
    """

    if method == 'mean':
        # 'mean' yöntemi: Sütunun ortalamasını al ve eksik değerleri bu değerle doldur
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        return df
    elif method == 'median':
        # 'median' yöntemi: Sütunun medyanını al ve eksik değerleri bu değerle doldur
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        return df
    elif method == 'mode':
        # 'mode' yöntemi: Sütunun modunu al ve eksik değerleri bu değerle doldur
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        return df
    elif method == 'value':
        # 'value' yöntemi: Kullanıcının belirttiği değerle eksik değerleri doldur
        df[col] = df[col].fillna(value)
        return df
    else:
        # Geçersiz yöntem belirtildiğinde hata yükselt
        raise ValueError("Only these options for method are allowed: ['mean', 'median', 'mode', 'value']")
