readme = """<div align="center">
  <h1>Loan Prediction Machine Learning Model</h1>
</div>

A Machine Learning classification project that predicts whether a loan application will be approved based on applicant and loan-related features.

## Objective

Build a classification model that can accurately predict loan approval status and handle class imbalance in the dataset.

## Dataset

- **Rows:** 45,000
- **Features:** 13 input features
- **Target:** `loan_status`
- **Test Size:** 30%

## Workflow

1. Load the loan dataset using Pandas.
2. Handle outliers using Winsorization.
3. Encode categorical features using Label Encoding.
4. Split the data into training and testing sets.
5. Scale features using Min-Max Scaling.
6. Handle class imbalance using SMOTE.
7. Train a Random Forest Classifier.
8. Evaluate the model using a classification report.
9. Save the trained model as `models/model.pkl`.

## Model

**Random Forest Classifier**

The model is trained on the preprocessed dataset and evaluated on unseen test data.

## Results

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 | 0.95 | 0.94 | 0.94 | 10,421 |
| 1 | 0.79 | 0.84 | 0.82 | 3,079 |

**Accuracy: 91%**

| Metric | Score |
|---|---:|
| Macro Avg Precision | 0.87 |
| Macro Avg Recall | 0.89 |
| Macro Avg F1-Score | 0.88 |
| Weighted Avg Precision | 0.92 |
| Weighted Avg Recall | 0.91 |
| Weighted Avg F1-Score | 0.92 |

## Project Structure

```text
Loan_Prediction_ML_Model/
│
├── data/
├── models/
├── research/
├── src/
│   └── loan_prediction_classification_model/
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       └── model_building.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md