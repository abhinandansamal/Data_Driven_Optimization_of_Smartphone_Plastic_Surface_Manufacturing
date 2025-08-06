from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.utils import save_object
from pathlib import Path

class ModelTrainer:
    """
    A class responsible for training a machine learning model (Random Forest Regressor)
    on the transformed training dataset and evaluating it on the test dataset.
    """
    def __init__(self):
        """
        Initializes the path to store the trained model.
        """
        self.model_path = Path("models/defect_rate_predictor.joblib")

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains a Random Forest Regressor model using the provided training and test data.
        Evaluates the model on the test data using R² score and saves the model if performance is acceptable.

        Args:
            train_array (np.ndarray): Numpy array of shape (n_samples, n_features + 1),
                                      where the last column is the target.
            test_array (np.ndarray): Numpy array of shape (n_samples, n_features + 1),
                                     where the last column is the target.

        Returns:
            float: The R² score of the model on the test data.

        Raises:
            Exception: If any error occurs during model training or evaluation.
        """
        print("Starting model training...")
        try:
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize Random Forest with tuned hyperparameters
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                max_features=1.0
            )

            # Train model
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)

            print(f"Best model (Random Forest) Test R2 Score: {test_r2:.4f}")

            # Save model only if performance is acceptable
            if test_r2 < 0.6:
                print("Model performance is below threshold. Not saving.")
            else:
                save_object(self.model_path, model)
                print("Model training completed.")

            return test_r2
        
        except Exception as e:
            raise Exception(f"Error during model training: {e}")