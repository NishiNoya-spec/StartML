### 6. Anova фильтрация  
 
from sklearn.feature_selection import f_classif 
 
def anova_filter(df, target="class", p_value_threshold=0.05): 
    df = df.copy() 
 
    X = df.drop(target, axis=1) 
    y = df[target] 
 
    numeric_columns, categorical_columns = numeric_categorical_columns(X) 
 
    initial_columns_count = X[numeric_columns].shape[1] 
 
    # Применяем ANOVA F-test 
    f_values, p_values = f_classif(X[numeric_columns], y) 
 
    # Создаем DataFrame с результатами 
    anova_results = pd.DataFrame({'Feature': X[numeric_columns].columns, 'F-Value': f_values, 'p-value': p_values}) 
    anova_results = anova_results.sort_values(by='p-value', ascending=True) 
     
    # Округляем значения p-value до 3 знаков после запятой 
    anova_results['p-value'] = anova_results['p-value'].round(3) 
 
    # Фильтруем значимые фичи (p-value < 0.05) 
    significant_features = anova_results[anova_results['p-value'] < p_value_threshold]['Feature'] 
    filtered_df = df[significant_features] 
 
    removed_unsignificant_cols_count = initial_columns_count - filtered_df.shape[1] 
 
    display(f"Кол-во удаленных незначимых фичей: {removed_unsignificant_cols_count}") 
 
    display("Результаты ANOVA F-теста:") 
    display(anova_results) 
 
    df = pd.concat([filtered_df, X[categorical_columns], y], axis=1) 
 
    return df, anova_results 
 
 
def plot_feature_importance(anova_results, n_features=20): 
    # Сортируем результаты по убыванию F-значений и выбираем первые n признаков 
    top_features = anova_results.sort_values(by='F-Value', ascending=False).head(n_features) 
 
    plt.figure(figsize=(12, 8)) 
    plt.barh(top_features['Feature'], top_features['F-Value'], color='skyblue') 
    plt.xlabel('F-Value') 
    plt.ylabel('Feature') 
    plt.title(f'Top {n_features} Features by F-Value') 
    plt.tight_layout() 
    plt.show()