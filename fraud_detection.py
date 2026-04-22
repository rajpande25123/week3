import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# Set visualization style
sns.set_theme(style="whitegrid")

# Create output directories for artifacts if they don't exist
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

def generate_data(n_samples=50000, weights=[0.99, 0.01]):
    """
    Generate an imbalanced synthetic dataset resembling credit card fraud.
    """
    print(f"Generating synthetic dataset with {n_samples} samples and {weights[1]*100}% fraud rate...")
    X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2,
                               n_informative=10, n_redundant=2, n_repeated=0,
                               weights=weights, random_state=42)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 21)])
    df['Class'] = y
    print("Dataset Generation Complete.\n")
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis and visualize class imbalances.
    """
    print("--- Exploratory Data Analysis ---")
    class_counts = df['Class'].value_counts()
    print("Class Distribution:")
    print(class_counts)
    print(f"Fraud Percentage: {class_counts[1] / len(df) * 100:.2f}%\n")

    # Visualize the class imbalance
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    plt.savefig("output/plots/class_distribution.png", dpi=300)
    plt.close()
    print("Saved 'class_distribution.png' in output/plots/")

def split_and_sample(df):
    """
    Split data and apply SMOTE to address class imbalance.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Set Size: {len(X_train)} | Test Set Size: {len(X_test)}")
    print(f"Training Frauds: {sum(y_train)} | Test Frauds: {sum(y_test)}\n")

    print("Applying SMOTE Oversampling on Training Data...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"New Training Set Size (After SMOTE): {len(X_train_sm)}")
    print(f"New Training Frauds (After SMOTE): {sum(y_train_sm)}\n")
    
    return X_train_sm, X_test, y_train_sm, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest classifier and output evaluation metrics.
    """
    print("--- Training Random Forest Classifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print("--- Model Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    
    auc_roc = roc_auc_score(y_test, y_scores)
    print(f"ROC-AUC Score: {auc_roc:.4f}\n")

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig("output/plots/confusion_matrix.png", dpi=300)
    plt.close()

    return y_test, y_scores

def plot_curves_and_tradeoffs(y_test, y_scores):
    """
    Plot PR Curve, ROC curve, and discuss thresholds.
    """
    print("--- Analyzing Precision-Recall Tradeoffs ---")

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_scores)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # PR Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("output/plots/roc_pr_curves.png", dpi=300)
    plt.close()
    print("Saved 'roc_pr_curves.png' in output/plots/\n")

    # Discuss tradeoffs
    threshold_options = [0.3, 0.5, 0.7]
    print("Business Decision Threshold Tradeoffs:")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Business Impact'}")
    print("-" * 70)
    
    with open("output/reports/business_tradeoffs.txt", "w") as f:
        f.write("Credit Card Fraud Detection - Business Decision Thresholds\n")
        f.write("="*60 + "\n\n")
        
        for thresh in threshold_options:
            # find closest threshold index
            idx = np.argmin(np.abs(thresholds_pr - thresh))
            p = precision[idx]
            r = recall[idx]
            
            if thresh == 0.3:
                impact = "High recall (catches ~most frauds), but blocks more innocent transactions (High False Positives)."
            elif thresh == 0.5:
                impact = "Balanced approach. Standard threshold."
            else:
                impact = "High precision (rarely blocks innocent people), but misses some frauds (High False Negatives)."
                
            print(f"{thresh:<10.1f} | {p:<10.3f} | {r:<10.3f} | {impact}")
            f.write(f"Threshold {thresh:.1f}:\n")
            f.write(f"- Precision: {p:.3f}\n")
            f.write(f"- Recall: {r:.3f}\n")
            f.write(f"- Strategy: {impact}\n\n")
    print("\nSaved detailed text report to 'output/reports/business_tradeoffs.txt'.")

def main():
    print("=" * 50)
    print("Credit Card Fraud Detection & Handling Imbalanced Data")
    print("=" * 50)
    
    df = generate_data()
    perform_eda(df)
    X_train_sm, X_test, y_train_sm, y_test = split_and_sample(df)
    y_test_real, y_scores = train_and_evaluate(X_train_sm, X_test, y_train_sm, y_test)
    plot_curves_and_tradeoffs(y_test_real, y_scores)

if __name__ == "__main__":
    main()
