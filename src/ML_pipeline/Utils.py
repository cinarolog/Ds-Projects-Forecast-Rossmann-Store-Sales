import pandas as pd
import numpy as np

def read_dataset(path):
    """
    Verilen dosya yolundan bir CSV dosyasını okuyan fonksiyon.

    Parametreler:
    - path: Okunacak CSV dosyasının yolu.

    Dönüş:
    - df: Okunan veri setini içeren Pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df

def merge_dataframes(df1, df2, col_name):
    """
    İki DataFrame'i belirtilen sütun adına göre birleştiren fonksiyon.

    Parametreler:
    - df1: Birinci DataFrame.
    - df2: İkinci DataFrame.
    - col_name: Birleştirme işlemi için kullanılacak sütun adı.

    Dönüş:
    - combined_data: Birleştirilmiş DataFrame.
    """
    combined_data = pd.merge(df1, df2, on=col_name)
    return combined_data

def remove_outliers(df, col, thresh):
    """
    Belirtilen sütundaki eşik değerinden büyük olan satırları kaldıran fonksiyon.

    Parametreler:
    - df: İşlem yapılacak DataFrame.
    - col: Outlier'ları kontrol etmek için seçilen sütun adı.
    - thresh: Eşik değeri, bu değerden büyük olan satırlar kaldırılacaktır.

    Dönüş:
    - df: Outlier'ları kaldırılmış DataFrame.
    """
    df = df.drop(df.loc[df[col] > thresh].index)
    return df

def year_from_date(df, date_col, new_col_name='year'):
    """
    Belirtilen tarih sütunundan yıl bilgisini çıkaran ve yeni bir sütun ekleyen fonksiyon.

    Parametreler:
    - df: İşlem yapılacak DataFrame.
    - date_col: Tarih bilgisini içeren sütun adı.
    - new_col_name: Eklenen yeni sütunun adı.

    Dönüş:
    - df: Yıl bilgisi eklenmiş DataFrame.
    """
    if new_col_name in df:
        raise KeyError(
            f"{new_col_name} column already exists. Please enter a different value for new_col_name")
    df[new_col_name] = pd.DatetimeIndex(df[date_col]).year
    return df

def month_from_date(df, date_col, new_col_name='month'):
    """
    Belirtilen tarih sütunundan ay bilgisini çıkaran ve yeni bir sütun ekleyen fonksiyon.

    Parametreler:
    - df: İşlem yapılacak DataFrame.
    - date_col: Tarih bilgisini içeren sütun adı.
    - new_col_name: Eklenen yeni sütunun adı.

    Dönüş:
    - df: Ay bilgisi eklenmiş DataFrame.
    """
    if new_col_name in df:
        raise KeyError(
            f"{new_col_name} column already exists. Please enter a different value for new_col_name")
    df[new_col_name] = pd.DatetimeIndex(df[date_col]).month
    return df

