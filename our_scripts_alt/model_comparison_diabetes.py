# %% Pakete
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid")

# %% Daten laden
df = pd.read_csv("diabetes.csv")
print("Shape:", df.shape)
print("\nKlassenverteilung (Outcome):")
print(df["Outcome"].value_counts())
print("\nErste 5 Zeilen:")
print(df.head())

# %% Features und Target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# %% Train/Test-Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Modell-Definitionen mit Hyperparameter-Grids
models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10, 100],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
    "SVM": {
        "estimator": CalibratedClassifierCV(
            estimator=SVC(random_state=42), ensemble=False
        ),
        "param_grid": {
            "estimator__C": [0.1, 1, 10],
            "estimator__kernel": ["rbf"],
            "estimator__gamma": ["scale", "auto"],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
        },
    },
}

# %% Cross-Validation + GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_estimators = {}
cv_results_summary = {}

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Trainiere: {name}")
    print(f"{'='*50}")

    grid = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_scaled, y_train)

    best_estimators[name] = grid.best_estimator_
    cv_results_summary[name] = {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
    }

    print(f"Beste Parameter: {grid.best_params_}")
    print(f"Bester CV ROC-AUC: {grid.best_score_:.4f}")

# %% Evaluierung auf dem Testsatz
metrics = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "ROC-AUC": [],
}
all_roc_data = {}

for name in best_estimators:
    model = best_estimators[name]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1-Score"].append(f1_score(y_test, y_pred))
    metrics["ROC-AUC"].append(roc_auc_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    all_roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

    print(f"\n{name} (beste Parameter: {cv_results_summary[name]['best_params']})")
    print(classification_report(y_test, y_pred, digits=4))

# %% Ergebnisse als DataFrame
results_df = pd.DataFrame(metrics, index=best_estimators.keys())
results_df["CV ROC-AUC"] = [cv_results_summary[n]["best_cv_score"] for n in best_estimators]
results_df["Beste Parameter"] = [str(cv_results_summary[n]["best_params"]) for n in best_estimators]

print("\n\n" + "="*60)
print("MODELLVERGLEICH - ZUSAMMENFASSUNG")
print("="*60)
print(results_df.to_string())

best_model_name = results_df["ROC-AUC"].idxmax()
print(f"\n{'*'*50}")
print(f"BESTES MODELL: {best_model_name}")
print(f"ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
print(f"Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"Beste Parameter: {cv_results_summary[best_model_name]['best_params']}")
print(f"{'*'*50}")

# %% Visualisierung 1: Metriken-Vergleich (Heatmap)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    results_df.drop(columns=["Beste Parameter"]).astype(float),
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Modellvergleich - Performance Metriken", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualisierung 2: Gruppierte Balken
fig, ax = plt.subplots(figsize=(12, 6))
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
plot_data = results_df[metric_cols]
x = np.arange(len(plot_data.index))
width = 0.15
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

for i, col in enumerate(metric_cols):
    bars = ax.bar(x + i * width, plot_data[col], width, label=col, color=colors[i])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

ax.set_xticks(x + width * 2)
ax.set_xticklabels(plot_data.index, rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("Modellvergleich - Alle Metriken", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualisierung 3: ROC-Kurven
fig, ax = plt.subplots(figsize=(9, 7))
for name, (fpr, tpr, auc) in all_roc_data.items():
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Zufallsrate")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC-Kurven - Vergleich aller Modelle", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig("roc_curves_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualisierung 4: Feature Importance (nur für baumbasierte Modelle)
tree_models = {
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
}
feature_names = X.columns

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, key) in zip(axes, tree_models.items()):
    if key in best_estimators:
        model = best_estimators[key]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax.barh(
                range(len(importances)),
                importances[indices],
                color="steelblue",
                edgecolor="black",
            )
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels(feature_names[indices])
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"{name}", fontweight="bold")
            ax.invert_yaxis()

plt.suptitle("Feature Importances (baum-basierte Modelle)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualisierung 5: Confusion Matrix des besten Modells
best_model = best_estimators[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    cbar=False,
    xticklabels=["Kein Diabetes", "Diabetes"],
    yticklabels=["Kein Diabetes", "Diabetes"],
    ax=ax,
)
ax.set_xlabel("Vorhersage", fontsize=11)
ax.set_ylabel("Wahrer Wert", fontsize=11)
ax.set_title(
    f"Confusion Matrix - Bestes Modell: {best_model_name}",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAlle Plots wurden gespeichert:")
print("- model_comparison_heatmap.png")
print("- model_comparison_bar.png")
print("- roc_curves_comparison.png")
print("- feature_importances.png")
print("- confusion_matrix_best_model.png")