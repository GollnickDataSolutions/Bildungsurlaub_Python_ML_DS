# %%
"""
Simplified Model Comparison Script
This script demonstrates the core concepts of comparing multiple ML models
on a classification task using the diabetes dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Outcome"].value_counts())

# Prepare features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split the data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with hyperparameter grids
models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {"C": [0.01, 0.1, 1, 10]},
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5]
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
        },
    },
    "SVM": {
        "estimator": CalibratedClassifierCV(
            estimator=SVC(random_state=42), ensemble=False
        ),
        "param_grid": {
            "estimator__C": [0.1, 1],
            "estimator__kernel": ["rbf"],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
    },
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
best_estimators = {}
results_summary = {}

print("Starting model comparison...")
print("="*50)

# Train and evaluate each model
for name, config in models.items():
    print(f"\nTraining: {name}")

    # Grid search with cross-validation
    grid = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,  # Reduced verbosity for cleaner output
    )
    grid.fit(X_train_scaled, y_train)

    # Store best model and results
    best_estimators[name] = grid.best_estimator_
    results_summary[name] = {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
    }

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

# Evaluate on test set
print("\n" + "="*50)
print("TEST SET EVALUATION")
print("="*50)

metrics = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "ROC-AUC": [],
}

for name in best_estimators:
    model = best_estimators[name]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1-Score"].append(f1_score(y_test, y_pred))
    metrics["ROC-AUC"].append(roc_auc_score(y_test, y_prob))

    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['Accuracy'][-1]:.4f}")
    print(f"  Precision: {metrics['Precision'][-1]:.4f}")
    print(f"  Recall: {metrics['Recall'][-1]:.4f}")
    print(f"  F1-Score: {metrics['F1-Score'][-1]:.4f}")
    print(f"  ROC-AUC: {metrics['ROC-AUC'][-1]:.4f}")

# Create summary DataFrame
results_df = pd.DataFrame(metrics, index=best_estimators.keys())
results_df["CV ROC-AUC"] = [results_summary[n]["best_cv_score"] for n in best_estimators]

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string())

# Find best model
best_model_name = results_df["ROC-AUC"].idxmax()
print(f"\n{'*'*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
print(f"Best parameters: {results_summary[best_model_name]['best_params']}")
print(f"{'*'*50}")

print("\nComparison complete!")