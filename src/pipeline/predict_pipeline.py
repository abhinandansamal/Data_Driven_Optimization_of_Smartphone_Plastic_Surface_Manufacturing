import pandas as pd
from src.utils import load_object
from pathlib import Path


class PredictPipeline:
    """
    Pipeline class for making predictions using a trained model and preprocessor.
    """
    def __init__(self):
        """
        Initializes the PredictPipeline by loading the model and preprocessor
        from the specified file paths.
        """
        self.model_path = Path("model/defect_rate_predictor.joblib")
        self.preprocessor_path = Path("model/preprocessor.joblib")
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)

    def predict(self, features):
        """
        Makes predictions on input features using the loaded model.
        
        Args:
            features (pd.DataFrame): Raw input features in a pandas DataFrame format.
        
        Returns:
            np.ndarray: Array containing the predicted defect rates.
        
        Raises:
            Exception: If any error occurs during preprocessing or prediction.
        """
        try:
            # Feature Engineering
            features["Energy_per_Unit"] = features["Energy Consumption (kWh)"] / (features["Batch Size (Units)"] + 1e-6)
            features["Process_Stress_Index"] = features["Temperature (°C)"] * features["Pressure (Pa)"]

            # Data Preprocessing
            processed_data = self.preprocessor.transform(features)

            # Construct column names for the transformed data
            ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            all_feature_names = (
                list(self.preprocessor.named_transformers_['skewed'].feature_names_in_) + \
                list(self.preprocessor.named_transformers_['symmetric'].feature_names_in_) + \
                list(ohe_feature_names)
            )

            # Create a DataFrame with processed features
            processed_df = pd.DataFrame(processed_data, columns=all_feature_names)

            # Drop high-VIF columns (as done during training)
            cols_to_drop = ['Process_Stress_Index', 'Pressure (Pa)']
            final_data = processed_df.drop(columns=cols_to_drop)

            # Make prediction
            prediction = self.model.predict(final_data)
            return prediction
        
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")
        

class CustomData:
    """
    A helper class to structure custom input data for prediction.
    """
    def __init__(self, **kwargs):
        """
        Initializes the CustomData instance with dynamic attributes
        based on the input keyword arguments.
        
        Args:
            **kwargs: Key-value pairs for feature names and their values.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data_as_dataframe(self):
        """
        Converts the stored attributes into a pandas DataFrame
        matching the expected input format for the prediction pipeline.
        
        Returns:
            pd.DataFrame: A single-row DataFrame containing the feature values.
        """
        data_dict = {
            "Temperature (°C)": [getattr(self, 'Temperature (°C)')],
            "Pressure (Pa)": [getattr(self, 'Pressure (Pa)')],
            "Cooling Rate (°C/min)": [getattr(self, 'Cooling Rate (°C/min)')],
            "Machine Speed (RPM)": [getattr(self, 'Machine Speed (RPM)')],
            "Raw Material Quality (Score)": [getattr(self, 'Raw Material Quality (Score)')],
            "Humidity (%)": [getattr(self, 'Humidity (%)')],
            "Ambient Temperature (°C)": [getattr(self, 'Ambient Temperature (°C)')],
            "Maintenance (Days Since)": [getattr(self, 'Maintenance (Days Since)')],
            "Operator Shift": [getattr(self, 'Operator Shift')],
            "Batch Size (Units)": [getattr(self, 'Batch Size (Units)')],
            "Energy Consumption (kWh)": [getattr(self, 'Energy Consumption (kWh)')],
            "Downtime (Minutes)": [getattr(self, 'Downtime (Minutes)')],
            "Production Line": [getattr(self, 'Production Line')]
        }

        return pd.DataFrame(data_dict)     