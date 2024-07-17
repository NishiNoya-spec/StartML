from sklearn.tree import DecisionTreeRegressor

base_pipe_DT = Pipeline([
    ("column_transformer", col_transform),
    ("decision_tree", DecisionTreeRegressor(random_state=2))
])

bagging_DT = BaggingRegressor(
    base_estimator=base_pipe_DT,
    n_estimators=100,
    random_state=1
)

bagging_DT.fit(X_train, y_train)

train_preds = bagging_DT.predict(X_train)
test_preds = bagging_DT.predict(X_test)

train_error = np.mean((train_preds - y_train)**2)
test_error = np.mean((test_preds - y_test)**2)

print(f"LR Качество на трейне: {train_error.round(3)}")
print(f"LR Качество на тесте: {test_error.round(3)}")