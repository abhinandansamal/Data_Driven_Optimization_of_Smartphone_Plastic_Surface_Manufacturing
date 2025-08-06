import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataIngestion:
    """
    Class responsible for ingesting raw data, saving it,
    and splitting it into training and testing datasets.
    """
    def __init__(self):
        """
        Initializes file paths for saving raw, training, and testing datasets.
        """
        self.raw_data_path = Path("data/raw_data.csv")
        self.train_data_path = Path("data/train.csv")
        self.test_data_path = Path("data/test.csv")

    def initiate_data_ingestion(self, data_path):
        """
        Loads the raw data from the given path, saves a copy, splits it into
        training and testing sets, and saves those to disk.

        Args:
            data_path (str): Path to the input CSV file containing raw data.

        Returns:
            Tuple[Path, Path]: Paths to the saved training and testing datasets.

        Raises:
            Exception: If any error occurs during data reading, splitting, or saving.
        """
        print("Starting data ingestion...")
        try:
            # Load raw data
            df = pd.read_csv(data_path)

            # Drop rows where the target is missing
            df.dropna(subset=['Defect Rate (%)'], inplace=True)

            # Create the artifacts directory if it doesn't exist
            self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing datasets
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            print("Data ingestion completed successfully.")
            return self.train_data_path, self.test_data_path
        
        except Exception as e:
            raise Exception(f"Error during data ingestion: {e}")