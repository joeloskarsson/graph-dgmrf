from sklearn.neural_network import MLPRegressor
import numpy as np

LAYER_CONFIGS = [
    (128,),
    (512,),
    (128,128,),
    (512,512,),
]
MAX_ITER = 10000

def mlp(X_train, y_train, X_test, config, **kwargs):
    # Do a quick hyperparameter tuning for layer configuration
    print("Tuning MLP layer config...")
    scores = []
    for layer_config in LAYER_CONFIGS:
        model = MLPRegressor(hidden_layer_sizes=layer_config, max_iter=MAX_ITER,
                early_stopping=True, validation_fraction=0.2, random_state=42)
        model.fit(X_train, y_train)

        val_score = model.best_validation_score_
        scores.append(val_score) # Validation score is R2-coeff.
        print("Layer config: {}, Epochs: {}, Score: {}".format(
            layer_config, model.n_iter_, val_score))

    best_index = np.argmax(scores)
    best_layer_config = LAYER_CONFIGS[best_index]
    print("Best layer config: {}".format(best_layer_config ))

    # Train and predict using ensemble
    print("Training and evaluating ensemble of {} models".format(config["n_ensemble"]))
    seeds = 43 + np.arange(config["n_ensemble"])
    predictions = []
    for seed in seeds:
        print("Model with seed {}".format(seed))
        model = MLPRegressor(hidden_layer_sizes=best_layer_config, max_iter=MAX_ITER,
                early_stopping=True, validation_fraction=0.2, random_state=seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions.append(pred)

    predictions = np.stack(predictions, axis=0) # Shape (N_ensemble, N_train)

    # Estimate predictive mean and std.-dev.
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    return pred_mean, pred_std

