
```python
# Beginning of optimization.py
# recent update Nov23 2024

import os
import json
from typing import Dict, Tuple
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from models import create_model
from utils import load_data, compute_complexity_factor

class HyperparameterOptimization:
    def __init__(self, config_file: str = 'config.json'):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: str) -> Dict:
        with open(config_file, 'r') as f:
            return json.load(f)

    def adjust_search_space(self, current_space: Dict, performance: float, threshold: float = 0.1) -> Dict:
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

    async def parallel_bayesian_optimization(self, X_train, y_train, X_val, y_val, n_iterations: int) -> Tuple[Dict, float, float]:
        """
        Performs parallel Bayesian optimization to find the best hyperparameters.
        
        Args:
            X_train (numpy.ndarray): The training data.
            y_train (numpy.ndarray): The training labels.
            X_val (numpy.ndarray): The validation data.
            y_val (numpy.ndarray): The validation labels.
            n_iterations (int): The number of optimization iterations.
        
        Returns:
            tuple: The best parameters, the best score, and the best quality score.
        """
        # Define the initial hyperparameter search space
        param_space = {
            'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(5, 15),
            'subsample': Real(0.5, 1.0),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
        }

        # Create the Gaussian Process Regressor
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10, random_state=self.config['random_seed'])

        # Create the Bayesian optimization object
        optimizer = BayesSearchCV(gpr, param_space, n_iter=n_iterations, random_state=self.config['random_seed'], n_jobs=self.config['num_parallel_jobs'])

        # Perform the Bayesian optimization
        optimizer.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Get the best hyperparameters
        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Compute the complexity factor
        complexity_factor = compute_complexity_factor(best_params)

        # Compute the quality score
        best_quality_score = 1 / complexity_factor

        return best_params, best_score, best_quality_score

    def perform_optimization(self, X_train, y_train, X_val, y_val):
        # Create the model
        model = create_model(self.config['model_type'], self.config['model_params'])

        # Perform the Bayesian optimization
        best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(X_train, y_train, X_val, y_val, self.config['num_optimization_steps'])

        # Adjust the search space based on the performance
        adjusted_search_space = self.adjust_search_space(param_space, best_score)

        # Perform another round of Bayesian optimization with the adjusted search space
        best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(X_train, y_train, X_val, y_val, self.config['num_optimization_steps'])

        # Save the best model
        best_model = create_model(self.config['model_type'], best_params)
        model_save_path = os.path.join(self.config['model_save_dir'], 'best_model.pkl')
        best_model.save(model_save_path)

        return best_params, best_score, best_quality_score


# End of optimization.py
```

