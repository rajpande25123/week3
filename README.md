# Credit-Card Fraud Detection

An end-to-end Machine Learning project to handle highly imbalanced data (resembling a Credit Card Fraud transaction dataset). 

This project explores an imbalanced dataset, applies data sampling techniques (SMOTE), trains a Random Forest Classifier, and evaluates the model beyond standard accuracy by focusing on Precision-Recall tradeoffs and ROC-AUC. 

## 📂 Project Structure

- `fraud_detection.py`: Main Python script containing the entire ML pipeline:
  - Generates a synthetic highly-imbalanced dataset (99% normal, 1% fraud).
  - Performs **EDA** and exports distribution plots to `output/plots/`.
  - Splits data and applies **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training set.
  - Trains a **Random Forest Classifier**.
  - Evaluates performance using **Precision, Recall, F1-Score, and ROC-AUC**.
  - Analyzes the **Precision-Recall framework** and provides a business-oriented threshold discussion.
- `requirements.txt`: Python package dependencies.
- `output/`: Generated output directories containing dataset visuals (ROC/PR curves, confusion matrices) and reports.

## 🚀 How to Run

1. Clone or download this project.
2. Ensure you have Python installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the model script:
   ```bash
   python fraud_detection.py
   ```
5. Check your console output for the classification reports and the `output/plots/` folder for visual plots!

## 📊 Business Tradeoffs: Precision vs Recall in Fraud Detection

In credit card fraud detection, accuracy is a misleading metric due to the massive class imbalance. If 99% of transactions are legitimate, a model that simply predicts "Not Fraud" for every transaction is 99% accurate, yet completely useless. Instead, we use **Precision** and **Recall**:

- **Precision**: Out of all the transactions the model flagged as fraud, how many were *actually* fraud?
  - *Business impact*: Low precision leads to high False Positives. Legitimate customers get their cards temporarily blocked, leading to a frustrating user experience and increased support volume.
- **Recall**: Out of all the *actual* fraud cases in the dataset, how many did the model successfully catch?
  - *Business impact*: Low recall leads to high False Negatives. Fraudsters succeed, costing the bank or merchant directly via chargebacks and monetary losses.

**The Decision Threshold Adjustment**:
By shifting the classification probability threshold, banks can tune their risk tolerance:
- **Threshold = 0.3**: Maximizes **Recall**. We catch almost all fraud but might accidentally decline more valid transactions. (Ideal for high-ticket purchases).
- **Threshold = 0.7**: Maximizes **Precision**. We only block transactions we are absolutely sure are fraud, providing a smoother experience for normal users but allowing some smaller fraud cases to slip through. (Ideal for automated micro-transactions).
