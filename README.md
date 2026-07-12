# Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave (churn) or remain with the bank using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The project includes complete data preprocessing, feature engineering, model building, training, evaluation, and visualization of training performance.

---

## 🎯 Objective

To build a binary classification model that accurately predicts customer churn based on customer demographic and banking information.

---

## 📂 Dataset

The project uses the **Churn_Modelling.csv** dataset.

### Input Features
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

### Target Variable
- **Exited**
  - 0 → Customer stays
  - 1 → Customer leaves

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Encode Categorical Variables
4. Split Dataset into Training and Testing Sets
5. Feature Scaling using StandardScaler
6. Build Artificial Neural Network
7. Train the Model
8. Apply Early Stopping to avoid overfitting
9. Predict Customer Churn
10. Evaluate Model Performance

---

## 🧠 ANN Architecture

- Input Layer
- Hidden Layer (ReLU)
- Dropout Layer
- Hidden Layer (ReLU)
- Dropout Layer
- Hidden Layer (ReLU)
- Output Layer (Sigmoid)

The model uses **Dropout** for regularization and **Early Stopping** to improve generalization and prevent overfitting.

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve

---

## 📁 Project Structure

```
Bank-Customer-Churn-Prediction/
│
├── ANN.ipynb
├── Churn_Modelling.csv
├── requirements.txt
├── README.md
└── images/
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── confusion_matrix.png
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Bank-Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Bank-Customer-Churn-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **ANN.ipynb** and run all cells.

---

## 📌 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Streamlit deployment
- Compare ANN with other machine learning models
- Explain predictions using SHAP or LIME

---

## 👩‍💻 Author

**Ishita Patel**
