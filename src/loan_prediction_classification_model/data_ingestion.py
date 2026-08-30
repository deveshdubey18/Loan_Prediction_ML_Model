import pandas as pd
import numpy as np

def data_ingestion():
  df = pd.read_csv(r'C:\Devesh ITV\Machine Learning\Loan_Prediction_Classification_Model\data\loan_data.csv')
  return df