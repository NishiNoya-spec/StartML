from sklearn.calibration import CalibratedClassifierCV

class PlattScalingCalibrator:
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrated_model = None

    def fit(self, X_train, y_train, method='sigmoid'):
        self.calibrated_model = CalibratedClassifierCV(self.base_model, method=method, cv='prefit')
        return self.calibrated_model.fit(X_train, y_train)

    def predict_proba(self, X):
        if not self.calibrated_model:
            raise ValueError("The model is not calibrated. Please fit the model first.")
        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        if not self.calibrated_model:
            raise ValueError("The model is not calibrated. Please fit the model first.")
        return self.calibrated_model.predict(X)

### *********************

#~ from library.data_processing.calibration_probs import PlattScalingCalibrator
#~ calibrator = PlattScalingCalibrator(pipe)
#~ pipe_calibrated = calibrator.fit(X_train, Y_train, method='isotonic')

### *********************