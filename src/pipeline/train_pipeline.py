from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    """
    A pipeline class that orchestrates the full model training process, including:
    - Data ingestion
    - Data transformation (preprocessing/feature engineering)
    - Model training
    """
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self, data_path):
        """
        Executes the training pipeline:
        1. Loads and splits raw data
        2. Transforms the data (preprocessing, feature engineering)
        3. Trains and evaluates the model

        Args:
            data_path (str): Path to the raw CSV dataset
        """
        train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion(data_path)
        train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        self.model_trainer.initiate_model_trainer(train_arr, test_arr)

if __name__ == "__main__":
    # Instantiate and run the training pipeline
    pipeline = TrainPipeline()
    # Provide the path to the raw dataset file
    pipeline.run("data/smart_phone_surface_plastic_manufacture.csv")