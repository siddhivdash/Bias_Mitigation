import pandas as pd

class DataProcessor:
    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)
    
    @staticmethod
    def save_csv(dataframe, file_path):
        dataframe.to_csv# filepath: src/utils/data_processing.py
import pandas as pd

class DataProcessor:
    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)
    
    @staticmethod
    def save_csv(dataframe, file_path):
        dataframe.to_csv