from sklearn.linear_model import BayesianRidge

def bayes_linreg(X_train, y_train, X_test, **kwargs):
    # Keep uninformative priors over alpha and lambda-parameters
    model = BayesianRidge(normalize=False)
    model.fit(X_train, y_train)

    pred_mean, pred_std = model.predict(X_test, return_std=True)
    return pred_mean, pred_std
