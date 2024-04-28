import pandas as pd

def print_clf(df_LR, df_DT, df_RF, df_LGBM, DUM_AUC, DUM_F1):
    """
    Функция вывода максимума F1 и гиперпараметров
    На вход берет таблицы значений F1,AUC-ROC по параметрам
    """
    print("DummyClassifier")
    print('Максимум AUC =', DUM_AUC)
    print('Максимум F1 =', DUM_F1)
    print()

    # будущая сводная таблица
    result_metrix = []

    print("LogisticRegression")
    print('Максимум F1 =', df_LR['f1'][df_LR['f1'].idxmax()])
    print(df_LR[df_LR['f1'] == df_LR['f1'].max()])
    result_metrix.append(
        ['LogisticRegression', df_LR['f1'][df_LR['f1'].idxmax()], df_LR['auc_roc'][df_LR['auc_roc'].idxmax()]])
    print()

    print("DecisionTreeClassifier")
    print('Максимум F1 = ', df_DT['f1'][df_DT['f1'].idxmax()])
    print(df_DT[df_DT['f1'] == df_DT['f1'].max()])
    result_metrix.append(
        ['DecisionTreeClassifier', df_DT['f1'][df_DT['f1'].idxmax()], df_DT['auc_roc'][df_DT['auc_roc'].idxmax()]])
    print()

    print("RandomForestClassifier")
    print('Максимум F1 =', df_RF['f1'][df_RF['f1'].idxmax()])
    print(df_RF[df_RF['f1'] == df_RF['f1'].max()])
    result_metrix.append(
        ['RandomForestClassifier', df_RF['f1'][df_RF['f1'].idxmax()], df_RF['auc_roc'][df_RF['auc_roc'].idxmax()]])

    print("LGBMClassifier")
    print('Максимум F1 =', df_LGBM['f1'][df_LGBM['f1'].idxmax()])
    print(df_LGBM[df_LGBM['f1'] == df_LGBM['f1'].max()])
    result_metrix.append(['LGBMClassifier', df_LGBM['f1'][df_LGBM['f1'].idxmax()],
                          df_LGBM['auc_roc'][df_LGBM['auc_roc'].idxmax()]])

    final_metrix = pd.DataFrame(result_metrix, columns=['Classifier', 'f1', 'auc_roc'])
    return final_metrix


#* выведем итог страданий кремния 
#~ pivot = print_clf(
#~    unbalanced_LR_data_metrix,
#~    unbalanced_DT_data_metrix,
#~    unbalanced_RF_data_metrix,
#~    unbalanced_LGBM_data_metrix,
#~    DUM_max_AUC,
#~    DUM_max_F1
#~ ) 
#~ pivot['balance_metod']='unbalanced'


#* DummyClassifier 
#* Максимум AUC = 0.5 
#* Максимум F1 = 0.0 
 
#* LogisticRegression 
#* Максимум F1 = 0.003 
#~       f1  auc_roc     x_c 
#~ 2  0.003    0.625     0.1 
#~ 3  0.003    0.627     1.0 
#~ 4  0.003    0.628    10.0 
#~ 5  0.003    0.628   100.0 
#~ 6  0.003    0.628  1000.0 
 
#* DecisionTreeClassifier 
#* Максимум F1 =  0.548 
#~      f1  auc_roc  depth 
#~ 8  0.548    0.755     20 
 
#* RandomForestClassifier 
#* Максимум F1 = 0.141 
#~        f1  auc_roc  depth  estim 
#~ 12  0.141    0.877     10     50 
#~ 13  0.141    0.886     10    100 
#~ 15  0.141    0.890     10    200 
#~ 16  0.141    0.893     10    250 
#~ 17  0.141    0.892     10    300 

#* LGBMClassifier 
#* Максимум F1 = 0.3745 
#~        f1  auc_roc  depth 
#~ 7  0.3745   0.9239     18 
#~ 8  0.3745   0.9239     20
