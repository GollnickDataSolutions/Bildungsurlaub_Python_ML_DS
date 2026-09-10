#%% Pakete
from typing import Any


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# %% Daten laden
file_path = "../001_Datasets/direct_marketing.csv"
banking = pd.read_csv(file_path)
banking


# %%
banking.describe()

#%%
banking.dtypes
# %%
banking["job"].unique()
# %% Kategorische Daten behandeln
banking_dummies = pd.get_dummies(banking, dtype=int, drop_first=True)

#%% TODO: Outlier-Detection

#%% in X, y trennen
X = banking_dummies.drop(columns=['y'])
y =banking_dummies[['y']]
print(f"banking_dummies shape: {banking_dummies.shape}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
#%% train-test split durchführen
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

#%% 
models = [LogisticRegression(), DummyClassifier(),  DecisionTreeClassifier(), RandomForestClassifier()]
df_results = pd.DataFrame({
    "model": models,
    "r2": [0.0 for i in range(len(models))]
})
df_results

#%% 
for i in range(len(df_results)):
    steps = [
        ('scaler', StandardScaler()),
        # ('log_reg', LogisticRegression())
        ('model', df_results.loc[i, 'model'])
    ]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    r2 = pipeline.score(X_test, y_test)
    df_results.loc[i, 'r2'] = r2

df_results

# Modell-Parameter ausgeben
# pipeline.named_steps['log_reg']

#%% 
# y_test_pred = pipeline.predict(X_test)

#%%


#%% Standardisierung durchführen
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled =scaler.transform(X_test)

#%% Was der Dummy-Classifier genau macht?
1 - np.sum(y_test["y"]) / len(y_test["y"])

#%% alternativ
from collections import Counter
Counter[Any](y_test["y"])

# %% Welche Merkmale sind besonders wichtig?
df_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': pipeline.named_steps["model"].feature_importances_
})
df_importances_job = df_importances[df_importances['feature'].str.startswith("job_")]
df_importances_job

#%% filter nach importance > 0.05
df_importances_threshold = df_importances[df_importances['importance']>0.05]


#%% Sortiere df_importances nach importance und nehme nur die Top 10 Merkmale
df_importances_top10 = df_importances.sort_values(by='importance', ascending=False).head(10)
df_importances_top10



# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(df_importances_top10, x='feature', y='importance')
plt.xticks(rotation=90)
plt.show()
# %% TODO: Confusionsmatrix darstellen
from sklearn.metrics import confusion_matrix
#%%
y_test_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels = ["No sale", "Sale"], yticklabels = ["No sale", "Sale"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# %%
import pandas as pd

# Beispiel-Daten für die Erstellung von zwei DataFrames mit den gleichen Spalten
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

# Diese beiden DataFrames können mit append/concat zusammengefügt werden:
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined
# %%
patient = X_test.iloc[101, :]
pipeline.predict_proba(patient)
# %%
