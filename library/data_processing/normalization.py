### 5. Нормализация данных  
 
from sklearn.preprocessing import StandardScaler 
 
def normalize_data(df, target="class"): 
    df = df.copy() 
     
    X = df.drop(target, axis=1) 
    y = df[target] 
     
    numeric_columns, categorical_columns = numeric_categorical_columns(X) 
     
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X[numeric_columns]) 
     
    X_scaled = pd.DataFrame(X_scaled, columns=numeric_columns, index=X.index) 
    df = pd.concat([X_scaled, X[categorical_columns], y], axis=1) 
     
    orig_numeric_columns = numeric_columns 
 
    return df, scaler, orig_numeric_columns 
 
 
def restore_original_values(df, scaler, orig_numeric_columns, target="class"): 
    df = df.copy() 
     
    X = df.drop(target, axis=1) 
    y = df[target] 
     
    numeric_columns, categorical_columns = numeric_categorical_columns(X) 
     
    # Определяем отсутствующие колонки 
    missing_columns = list(set(orig_numeric_columns) - set(numeric_columns)) 
     
    # Создаем DataFrame с отсутствующими колонками, заполненными нулями 
    missing_df = pd.DataFrame(0, index=X.index, columns=missing_columns) 
     
    # Добавляем отсутствующие колонки к X 
    X = pd.concat([X, missing_df], axis=1) 
     
    # Применяем обратное преобразование нормализованных данных 
    X_restored = pd.DataFrame(scaler.inverse_transform(X[orig_numeric_columns]), columns=orig_numeric_columns, index=df.index) 
     
    # Удаляем добавленные нулевые колонки 
    X_restored = X_restored.drop(missing_columns, axis=1) 
     
    # Восстанавливаем оригинальный DataFrame, заменяя нормализованные колонки на восстановленные 
    df_restored = pd.concat([X_restored, df.drop(numeric_columns, axis=1)], axis=1) 
     
    return df_restored