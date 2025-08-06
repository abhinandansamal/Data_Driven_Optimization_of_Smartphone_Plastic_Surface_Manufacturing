import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from src.utils import save_object

class DataTransformation:
    """
    Handles data transformation tasks including feature engineering,
    preprocessing (scaling, encoding), and saving the transformation pipeline.
    """
    def __init__(self):
        """
        Initializes the path for saving the fitted preprocessing object.
        """
        self.preprocessor_path = Path("models/preprocessor.joblib")

    def get_data_transformer_object(self):
        """
        Constructs a preprocessing pipeline for skewed numerical, symmetric numerical,
        and categorical features.

        Returns:
            ColumnTransformer: A composed transformer with specific preprocessing logic.

        Raises:
            Exception: If an error occurs while creating the transformer.
        """
        try:
            # Define feature groups
            skewed_features = [
                "Temperature (°C)", "Pressure (Pa)", "Maintenance (Days Since)",
                "Batch Size (Units)", "Downtime (Minutes)", "Process_Stress_Index"
            ]
            symmetric_features = [
                "Cooling Rate (°C/min)", "Machine Speed (RPM)", "Raw Material Quality (Score)",
                "Humidity (%)", "Ambient Temperature (°C)", "Energy Consumption (kWh)", "Energy_per_Unit"
            ]
            categorical_features = [
                "Operator Shift", "Production Line"
            ]

            # Pipelines for different feature types
            log_transformer = FunctionTransformer(np.log1p)
            skewed_pipeline = Pipeline(steps=[
                ("log", log_transformer),
                ("scale", StandardScaler())
            ])
            symmetric_pipeline = Pipeline(steps=[
                ("scale", StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # Combine into a single column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("skewed", skewed_pipeline, skewed_features),
                    ("symmetric", symmetric_pipeline, symmetric_features),
                    ("cat", categorical_pipeline, categorical_features)
                ],
                remainder="passthrough"
            )
            return preprocessor
        except Exception as e:
            raise Exception(f"Error creating data transformer: {e}")
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing to the training and testing datasets, including feature
        engineering, transformation, and saving the preprocessor.

        Args:
            train_path (str): Path to the training dataset CSV.
            test_path (str): Path to the testing dataset CSV.

        Returns:
            tuple: (transformed training array, transformed testing array, path to saved preprocessor)

        Raises:
            Exception: If any step in the transformation process fails.
        """
        print("Starting data transformation...")
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Define target and drop columns
            target_column = "Defect Rate (%)"
            drop_columns = [target_column, "Production Output (Units)", "Unnamed: 0", "Production Run ID", "Date"]

            input_feature_train_df = train_df.drop(columns=drop_columns, errors="ignore")
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns, errors="ignore")
            target_feature_test_df = test_df[target_column]

            # Handle missing values and engineer features
            for df in [input_feature_train_df, input_feature_test_df]:
                numeric_cols = df.select_dtypes(include=np.number).columns
                imputer = SimpleImputer(strategy="median")
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                df["Energy_per_Unit"] = df["Energy Consumption (kWh)"] / (df["Batch Size (Units)"] + 1e-6)
                df["Process_Stress_Index"] = df["Temperature (°C)"] * df["Pressure (Pa)"]

            # Get preprocessor and apply it
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Reconstruct DataFrames for column dropping
            ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
            all_feature_names = list(preprocessor.named_transformers_["skewed"].feature_names_in_) + \
                                list(preprocessor.named_transformers_["symmetric"].feature_names_in_) + \
                                list(ohe_feature_names)
            
            train_processed_df = pd.DataFrame(input_feature_train_arr, columns=all_feature_names)
            test_processed_df = pd.DataFrame(input_feature_test_arr, columns=all_feature_names)

            # Drop high VIF columns
            cols_to_drop = ["Process_Stress_Index", "Pressure (Pa)"]
            train_final = train_processed_df.drop(columns=cols_to_drop)
            test_final = test_processed_df.drop(columns=cols_to_drop)

            # Combine features and target
            train_arr = np.c_[train_final.to_numpy(), np.array(target_feature_train_df)]
            test_arr = np.c_[test_final.to_numpy(), np.array(target_feature_test_df)]

            # Save the preprocessor for inference
            save_object(self.preprocessor_path, preprocessor)
            print("Data transformation completed.")

            return train_arr, test_arr, self.preprocessor_path
        
        except Exception as e:
            raise Exception(f"Error during data transformation: {e}")