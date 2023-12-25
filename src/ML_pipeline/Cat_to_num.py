import pandas as pd
import numpy as np
from sklearn import preprocessing

def cat_to_num(df, col, method='default', values=None):
    """
    Kategorik verileri sayısal verilere dönüştüren fonksiyon.

    Parametreler:
    - df: Pandas DataFrame, dönüşüm yapılacak veri seti.
    - col: str, dönüşüm yapılacak sütun adı.
    - method: str, dönüşüm yöntemi ('default' veya 'custom').
    - values: dict, 'custom' yöntemi için özel dönüşüm eşleştirmelerini içeren sözlük.

    Dönüş:
    - df: Dönüştürülmüş DataFrame.
    """

    if method == 'default':
        # 'default' yöntemi: LabelEncoder kullanarak otomatik dönüşüm yap
        label_encoder = preprocessing.LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        return df
    elif method == 'custom':
        # 'custom' yöntemi: Kullanıcının belirttiği özel eşleştirmeleri uygula
        for key, val in values.items():
            df[col].loc[df[col] == key] = val
        return df
    else:
        # Geçersiz yöntem belirtildiğinde hata yükselt
        raise ValueError("Only these options for method are allowed: ['default', 'custom']")
