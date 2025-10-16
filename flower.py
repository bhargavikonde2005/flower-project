# fraud_detection.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# imbalance tools
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

def load_data_or_synthetic(path='creditcard.csv', n_samples=100000):
    """
    Try to load real Kaggle dataset (creditcard.csv). If not present,
    create a synthetic, realistic imbalanced dataset using sklearn.
    """
    if os.path.exists(path):
        print(f"Loading dataset from {path} ...")
        df = pd.read_csv(path)
        # Kaggle dataset uses 'Class' column where 1 = fraud, 0 = legit
        if 'Class' not in df.columns:
            raise ValueError("Expected 'Class' column in CSV.")
        return df
    else:
        # Create synthetic dataset
        print(f"{path} not found — creating synthetic imbalanced dataset (for demo).")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=30,
            n_informative=6,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            weights=[0.995, 0.005],  # heavy imbalance ~0.5% fraud
            flip_y=0.001,
            class_sep=1.0,
            random_state=RANDOM_STATE
        )
        cols = [f'V{i}' for i in range(1, X.shape[1]+1)]
        df = pd.DataFrame(X, columns=cols)
        df['Amount'] = np.random.exponential(scale=50, size=n_samples)  # synthetic amounts
        df['Class'] = y
        return df

def preprocess(df):
    """
    Basic preprocess:
    - separate X / y
    - scale numeric features
    - return arrays and scaler
    """
    # if real Kaggle dataset, it already has many PCA features V1..V28, 'Time', 'Amount', 'Class'
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must have 'Class' column")

    y = df['Class'].values
    X = df.drop(columns=['Class']).copy()

    # Drop 'Time' if present (often not useful)
    if 'Time' in X.columns:
        X = X.drop(columns=['Time'])

    # Standard scale numeric features (fit on training later in pipeline)
    feature_names = X.columns.tolist()
    return X, y, feature_names

def evaluate_model(name, model, X_test, y_test):
    """
    Print classic metrics and return useful metrics dict.
    """
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    y_pred = model.predict(X_test)

    print(f"\n--- Results: {name} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR AUC): {avg_precision:.4f}")

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }

def plot_curves(metrics_dict, title_suffix=""):
    y_test = metrics_dict["y_test"]
    y_proba = metrics_dict["y_proba"]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {metrics_dict['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"PR curve (AP = {metrics_dict['avg_precision']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 1. Load data
    df = load_data_or_synthetic('creditcard.csv', n_samples=100000)

    # quick class distribution
    print("\nClass distribution (value:count):")
    print(df['Class'].value_counts())

    # 2. Preprocess
    X_df, y, feature_names = preprocess(df)

    # 3. Train/test split (stratified)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {X_train_df.shape[0]}, Test size: {X_test_df.shape[0]}")

    # 4. Option A: Baseline model without sampling
    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    print("\nTraining baseline Logistic Regression (class_weight='balanced') ...")
    baseline_pipe.fit(X_train_df, y_train)
    baseline_metrics = evaluate_model("Baseline LogisticRegression (balanced)", baseline_pipe, X_test_df, y_test)
    plot_curves(baseline_metrics, title_suffix="(Baseline)")

    # 5. Option B: Oversampling with SMOTE + Logistic
    print("\nTraining with SMOTE oversampling + LogisticRegression ...")
    smote_pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, n_jobs=-1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    smote_pipeline.fit(X_train_df, y_train)
    smote_metrics = evaluate_model("SMOTE + LogisticRegression", smote_pipeline, X_test_df, y_test)
    plot_curves(smote_metrics, title_suffix="(SMOTE + Logistic)")

    # 6. Option C: Undersampling + RandomForest
    print("\nTraining with RandomUnderSampler undersampling + RandomForest ...")
    rus_pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("rus", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))
    ])
    rus_pipeline.fit(X_train_df, y_train)
    rus_metrics = evaluate_model("RUS + RandomForest", rus_pipeline, X_test_df, y_test)
    plot_curves(rus_metrics, title_suffix="(RUS + RF)")

    # 7. Option D: Hyperparameter tuning example (RandomForest small grid)
    print("\nOptional: Quick hyperparameter search for RandomForest (on undersampled data). This may take time.")
    try:
        # create smaller undersampled train for GridSearch (to speed up)
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_rus, y_rus = rus.fit_resample(X_train_df, y_train)
        scaler = StandardScaler().fit(X_rus)
        X_rus_scaled = scaler.transform(X_rus)

        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [6, 12, None],
            "min_samples_split": [2, 5]
        }
        gs = GridSearchCV(rf, param_grid, scoring='f1', cv=3, n_jobs=-1)
        gs.fit(X_rus_scaled, y_rus)
        print("Best params:", gs.best_params_)

        # evaluate best estimator on full test set (scale test with scaler)
        best_rf = gs.best_estimator_
        # need to build pipeline to scale test features
        from sklearn.pipeline import make_pipeline
        final_pipe = make_pipeline(StandardScaler(), best_rf)
        final_pipe.fit(X_rus, y_rus)  # fit on undersampled scaled data
        final_metrics = evaluate_model("Tuned RF (undersampled)", final_pipe, X_test_df, y_test)
        plot_curves(final_metrics, title_suffix="(Tuned RF)")
    except Exception as e:
        print("Skipping hyperparameter search due to:", e)

    print("\nDone. Compare the metrics above (Precision, Recall, F1, ROC AUC, PR AUC) and choose the approach that best fits your risk tolerance (fraud detection usually prioritizes recall and precision on the fraud class).")

if __name__ == "__main__":
    main()
