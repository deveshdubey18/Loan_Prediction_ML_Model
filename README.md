# 🏦 Loan Prediction Machine Learning Model

> A Machine Learning classification project that predicts whether a loan application will be **approved or rejected** based on applicant and loan-related features.

---

## 📌 Project Overview

Loan approval is an important decision for financial institutions. This project uses **Machine Learning classification techniques** to predict the loan approval status of an applicant.

The project follows a **modular machine learning approach**, separating data ingestion, preprocessing, and model building into individual modules.

The final model uses a **Random Forest Classifier** and achieves approximately **91% accuracy** on unseen test data.

---

## 🎯 Objective

The main objectives of this project are:

* Predict whether a loan application will be approved.
* Handle numerical outliers effectively.
* Convert categorical data into numerical form.
* Scale numerical features.
* Handle class imbalance using **SMOTE**.
* Train and evaluate a classification model.
* Save the trained model for future use.

---

## 📊 Dataset

The project uses a loan application dataset containing **45,000 records** and **13 input features**.

| Dataset Information | Details                  |
| ------------------- | ------------------------ |
| 📁 Dataset          | `loan_data.csv`          |
| 📈 Records          | 45,000                   |
| 🔢 Input Features   | 13                       |
| 🎯 Target Variable  | `loan_status`            |
| 🧪 Test Size        | 30%                      |
| 🌳 Model            | Random Forest Classifier |

---

## 🔄 Machine Learning Workflow

The complete workflow of the project is:

```text
              📂 Dataset
                  │
                  ▼
          📥 Data Ingestion
                  │
                  ▼
        🧹 Data Preprocessing
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   Outlier     Encoding   Scaling
   Handling    Categorical   │
        │         │         │
        └─────────┼─────────┘
                  ▼
              SMOTE
                  │
                  ▼
          Train/Test Split
                  │
                  ▼
       🌳 Random Forest Model
                  │
                  ▼
            📊 Evaluation
                  │
                  ▼
          💾 Save Model
```

---

## 🧹 Data Preprocessing

Several preprocessing techniques are applied before training the model.

### 1. Outlier Handling

**Winsorization** is used to cap extreme values in numerical columns.

This helps reduce the influence of extreme observations without completely removing them from the dataset.

### 2. Categorical Encoding

Categorical columns are converted into numerical values using **Label Encoding**.

### 3. Train-Test Split

The dataset is divided into:

* **70% Training Data**
* **30% Testing Data**

The testing data is kept unseen during model training to evaluate the model's performance.

### 4. Feature Scaling

**Min-Max Scaling** is applied to transform the features into a common numerical range.

### 5. Handling Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data to balance the classes.

This helps the model perform better when the target classes are imbalanced.

---

## 🤖 Machine Learning Model

### 🌳 Random Forest Classifier

The project uses a **Random Forest Classifier** for loan approval prediction.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a more reliable prediction.

### Why Random Forest?

* 🌳 Handles non-linear relationships
* 📊 Works well with different types of features
* 🛡️ Reduces overfitting compared to a single decision tree
* ⚡ Provides strong classification performance
* 🔄 Works well with relatively large datasets

---

## 📈 Model Performance

The model achieved approximately **91% accuracy** on the test dataset.

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----: | --------: | -----: | -------: |
|     0 |      0.95 |   0.94 |     0.94 |
|     1 |      0.79 |   0.84 |     0.82 |

### Overall Metrics

| Metric                 |   Score |
| ---------------------- | ------: |
| 🎯 Accuracy            | **91%** |
| Macro Avg Precision    |    0.87 |
| Macro Avg Recall       |    0.89 |
| Macro Avg F1-Score     |    0.88 |
| Weighted Avg Precision |    0.92 |
| Weighted Avg Recall    |    0.91 |
| Weighted Avg F1-Score  |    0.92 |

---

## 🧩 Modular Project Structure

The project follows a modular architecture to keep different stages of the machine learning pipeline organized.

```text
Loan_Prediction_ML_Model/
│
├── 📂 data/
│   └── loan_data.csv
│
├── 📂 models/
│   └── model.pkl
│
├── 📂 research/
│   ├── Experiment1_MLFlow.ipynb
│   └── Experiment2_AutoGluon.ipynb
│
├── 📂 src/
│   └── loan_prediction_classification_model/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       └── model_building.py
│
├── main.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## ⚙️ Modules

### 📥 Data Ingestion

**File:** `data_ingestion.py`

Responsible for loading the loan dataset using Pandas and returning it as a DataFrame.

### 🧹 Data Preprocessing

**File:** `data_preprocessing.py`

Handles:

* Duplicate handling
* Numerical and categorical feature separation
* Outlier treatment
* Label Encoding
* Train-test splitting
* Min-Max Scaling
* SMOTE

### 🤖 Model Building

**File:** `model_building.py`

Responsible for:

* Creating the Random Forest model
* Training the model
* Generating predictions
* Creating the classification report
* Saving the trained model as `model.pkl`

### 🚀 Main Pipeline

**File:** `main.py`

The main file connects all modules and executes the complete machine learning pipeline.

```text
Data Ingestion
      ↓
Preprocessing
      ↓
Model Training
      ↓
Prediction
      ↓
Evaluation
      ↓
Model Saving
```

---

## 🛠️ Technologies & Libraries

### Programming Language

🐍 **Python 3.13+**

### Libraries

* 🐼 **Pandas** — Data manipulation and analysis
* 🔢 **NumPy** — Numerical operations
* 🤖 **Scikit-learn** — Machine learning algorithms and preprocessing
* ⚖️ **Imbalanced-learn** — SMOTE for class balancing
* 📊 **Matplotlib** — Data visualization
* 📈 **Seaborn** — Statistical visualization
* 🔬 **FLAML** — Automated machine learning experimentation
* 🔥 **PyTorch** — Machine learning framework

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/deveshdubey18/Loan_Prediction_ML_Model.git
```

### 2️⃣ Navigate to the Project

```bash
cd Loan_Prediction_ML_Model
```

### 3️⃣ Install Dependencies

```bash
pip install -e .
```

### 4️⃣ Run the Project

```bash
python main.py
```

---

## 💾 Model Output

After successful execution, the trained model is saved as:

```text
models/model.pkl
```

The application also prints the dataset shape and classification report in the terminal.

---

## 🔬 Research & Experiments

The repository also contains experimentation notebooks:

* `Experiment1_MLFlow.ipynb` — MLflow-based experimentation
* `Experiment2_AutoGluon.ipynb` — AutoML experimentation using AutoGluon

These experiments were used for exploring and comparing different machine learning approaches.

---

## 🌟 Key Highlights

* ✅ End-to-end machine learning classification project
* ✅ Modular project architecture
* ✅ Outlier treatment using Winsorization
* ✅ Categorical feature encoding
* ✅ Feature scaling using Min-Max Scaler
* ✅ Class imbalance handling using SMOTE
* ✅ Random Forest classification
* ✅ ~91% test accuracy
* ✅ Model serialization using Pickle
* ✅ Research experiments with MLflow and AutoML

---

## 📌 Future Improvements

Some possible improvements for the project include:

* 🔹 Hyperparameter tuning for Random Forest
* 🔹 Feature importance analysis
* 🔹 Cross-validation
* 🔹 Model comparison with other classifiers
* 🔹 Improved data validation
* 🔹 Deployment using Streamlit or Flask
* 🔹 Creating an API for real-time loan predictions

---

## 👨‍💻 Author

**Devesh Dubey**

🔗 GitHub: [@deveshdubey18](https://github.com/deveshdubey18)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub!

**Thank you for visiting! 🚀**
