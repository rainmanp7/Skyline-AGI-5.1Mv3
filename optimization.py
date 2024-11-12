
```python
# Beginning of optimization.py
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Define hyperparameter space for Bayesian optimization
param_space = {
    'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 15),
    'subsample': Real(0.5, 1.0),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 10),
}

# Define a function to adjust the search space
def adjust_search_space(current_space, performance, threshold=0.1):
    """
    Adjusts the search space based on the performance of the model.
    If the performance is above the threshold, the learning rate range is increased.
    If the performance is below the threshold, the learning rate range is decreased.
    """
    adjusted_space = current_space.copy()
    if performance > threshold:
        adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low * 0.1, 
                                               current_space['learning_rate'].high * 10, 
                                               prior='log-uniform')
    else:
        adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low, 
                                               current_space['learning_rate'].high * 0.1, 
                                               prior='log-uniform')
    return adjusted_space

async def parallel_bayesian_optimization(initial_param_space, X_train, y_train, X_test, y_test, n_iterations, complexity_factor):
    """
    Performs parallel Bayesian optimization to find the best hyperparameters.
    
    Args:
        initial_param_space (dict): The initial hyperparameter search space.
        X_train (numpy.ndarray): The training data.
        y_train (numpy.ndarray): The training labels.
        X_test (numpy.ndarray): The test data.
        y_test (numpy.ndarray): The test labels.
        n_iterations (int): The number of optimization iterations.
        complexity_factor (float): The complexity factor to compute the quality score.
    
    Returns:
        tuple: The best parameters, the best score, and the best quality score.
    """
    # Existing optimization logic
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10, random_state=42)
    optimizer = BayesSearchCV(gpr, initial_param_space, n_iter=n_iterations, random_state=42, n_jobs=-1)
    optimizer.fit(X_train, y_train)
    best_params = optimizer.best_params_
    best_score = optimizer.best_score_

    # Compute the quality score based on the complexity factor
    best_quality_score = 1 / complexity_factor

    return best_params, best_score, best_quality_score

# End of optimization.py
```
