from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

pipe_dt = Pipeline([
    ("column_transformer", col_transform),
    ("decision_tree", DecisionTreeRegressor())
])

pipe_lr = Pipeline([
    ("column_transformer", col_transform),            
    ("Lasso", Lasso())
])

pipe_knn = Pipeline([
    ("column_transformer", col_transform),
    ("knn", KNeighborsRegressor())
])
 
estimators = [
    ("dt", pipe_dt),
    ("lr", pipe_lr),
    ("knn", pipe_knn)
]

final_estimator = DecisionTreeRegressor(max_depth=3)

### Base learners are fitted on the full X
### while the final estimator is trained
### using cross-validated predictions of the base learners
### using cross_val_predict.

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=2
)

stacking_model.fit(X_train, y_train)

### Замерим качество работы такой модели
### Возьмем MSLE

train_preds = stacking_model.predict(X_train)
test_preds = stacking_model.predict(X_test)

train_error = np.mean((train_preds - y_train)**2)
test_error = np.mean((test_preds - y_test)**2)

print(f"Качество на трейне: {train_error.round(3)}")
print(f"Качество на тесте: {test_error.round(3)}")