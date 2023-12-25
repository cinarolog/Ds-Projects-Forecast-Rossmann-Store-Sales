# Belirli bir regresyon algoritması için parametre ızgarasını seçen fonksiyon
def choose_param_grid(param_grid):
    # Eğer regresyon algoritması doğrusal regresyon ise:
    if param_grid == 'linear_reg':
        # Normalize ve fit_intercept parametrelerini içeren bir sözlük döndür
        return {'normalize': [True, False], 'fit_intercept': [True, False]}
    
    # Eğer regresyon algoritması Stokastik Gradyan İnişli regresyon ise:
    elif param_grid == 'SGD_reg':
        # Alpha, max_iter, loss, penalty ve tol parametrelerini içeren bir sözlük döndür
        return {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'max_iter': [1000, 5000, 10000],
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'tol': [1e-3, 1e-4, 1e-5]
        }
    
    # Eğer regresyon algoritması Rastgele Orman regresyon ise:
    elif param_grid == 'RF_reg':
        # n_estimators, max_depth, min_samples_split ve min_samples_leaf parametrelerini içeren bir sözlük döndür
        return {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Eğer regresyon algoritması Karar Ağacı regresyon ise:
    elif param_grid == 'dtree_reg':
        # max_depth, min_samples_split ve min_samples_leaf parametrelerini içeren bir sözlük döndür
        return {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Eğer regresyon algoritması Ridge regresyon ise:
    elif param_grid == 'ridge':
        # alpha ve normalize parametrelerini içeren bir sözlük döndür
        return {'alpha': [0.01, 0.1, 1, 10, 100], 'normalize': [True, False]}
    
    # Eğer regresyon algoritması Lasso regresyon ise:
    elif param_grid == 'lasso':
        # alpha ve normalize parametrelerini içeren bir sözlük döndür
        return {'alpha': [0.01, 0.1, 1, 10, 100], 'normalize': [True, False]}
    
    # Eğer regresyon algoritması Gradient Boosting regresyon ise:
    elif param_grid == 'gb_reg':
        # n_estimators, max_depth, learning_rate ve subsample parametrelerini içeren bir sözlük döndür
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    # Eğer regresyon algoritması k-En Yakın Komşu regresyon ise:
    elif param_grid == 'knn_reg':
        # n_neighbors, weights ve p parametrelerini içeren bir sözlük döndür
        return {
            'n_neighbors': [3, 5, 10, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3]
        }
    
    # Eğer regresyon algoritması Destek Vektör Makineleri regresyon ise:
    elif param_grid == 'svm_reg':
        # C, kernel, gamma ve tol parametrelerini içeren bir sözlük döndür
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'tol': [1e-3, 1e-4, 1e-5]
        }
    
    # Eğer regresyon algoritması Elastic Net regresyon ise:
    elif param_grid == 'elastic_net':
        # alpha, l1_ratio ve normalize parametrelerini içeren bir sözlük döndür
        return {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.5, 0.9],
            'normalize': [True, False]
        }
    
    # Eğer regresyon algoritması Huber regresyon ise:
    elif param_grid == 'huber_reg':
        # epsilon ve max_iter parametrelerini içeren bir sözlük döndür
        return {
            'epsilon': [1.1, 1.35, 1.5, 2.0],
            'max_iter': [100, 500, 1000, 2000]
        }
    
    # Geçersiz bir parametre ızgara anahtarı verilirse ValueError hatası fırlat
    else:
        raise ValueError(f"Geçersiz param_grid: {param_grid}")
