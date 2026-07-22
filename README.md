# Customer Churn Prediction using Artificial Neural Networks

An end-to-end machine learning project that predicts whether a bank customer is likely to churn (leave the bank), built using an Artificial Neural Network (ANN) with TensorFlow/Keras, and deployed as an interactive Streamlit web app.

## Overview

Customer churn is a critical problem for banks — retaining an existing customer is significantly cheaper than acquiring a new one. This project builds a full pipeline that takes raw customer data, processes it, trains a neural network classifier, and serves real-time predictions through a web interface.

## Dataset

- **Source:** Churn_Modelling.csv (10,000 bank customer records)
- **Target variable:** `Exited` (1 = churned, 0 = stayed)
- **Class distribution:** ~80% stayed, ~20% churned (imbalanced dataset — handled explicitly during evaluation)

**Features used:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.

## Project Workflow

1. **Data Exploration** — verified data integrity (no missing values), analyzed class balance and feature distributions
2. **Data Cleaning** — removed non-predictive identifier columns (RowNumber, CustomerId, Surname)
3. **Feature Encoding** — Label Encoding for Gender, One-Hot Encoding for Geography
4. **Train/Test Split** — 80/20 split performed *before* scaling to prevent data leakage
5. **Feature Scaling** — StandardScaler fit on training data only, applied to test data
6. **Model Architecture** — Sequential ANN (12 → 64 → 32 → 1) with ReLU hidden layers and Sigmoid output
7. **Training** — Adam optimizer, binary crossentropy loss, EarlyStopping with validation split to prevent overfitting
8. **Evaluation** — assessed using accuracy, confusion matrix, precision, recall, and F1-score (not accuracy alone, due to class imbalance)
9. **Deployment** — trained model, encoders, and scaler serialized and served via a Streamlit web app

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 86.4% |
| Precision (Churn class) | 0.74 |
| Recall (Churn class) | 0.47 |
| F1-score (Churn class) | 0.57 |

**Key finding:** while overall accuracy was strong, recall on the churned class was notably lower — a direct consequence of the dataset's class imbalance (80/20 split). This was identified through confusion matrix analysis rather than relying on accuracy alone. Potential improvements include class weighting, oversampling (SMOTE), and decision threshold tuning.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (preprocessing, encoders, evaluation metrics)
- TensorFlow / Keras (neural network)
- Streamlit (web app deployment)

## Project Structure

```
churn-prediction-ann/
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   └── churn_project.ipynb
├── models/
│   ├── churn_model.h5
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   └── scaler.pkl
├── churn_app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/op-git29/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run churn_app.py
```

### 4. Or explore the notebook
```bash
jupyter notebook notebooks/churn_project.ipynb
```

## Future Improvements

- Address class imbalance using class weighting or SMOTE
- Hyperparameter tuning (layers, neurons, learning rate) via GridSearchCV/KerasTuner
- Add SHAP/feature importance analysis for model interpretability
- Deploy to cloud (Streamlit Community Cloud / Render / HuggingFace Spaces)

## Author

Built as a hands-on learning project to understand the complete ML pipeline — from raw data to deployed model.
