#%% Pakete
import pandas as pd

#%% Datenimport
banking = pd.read_csv("direct_marketing.csv")
#%% Explorative Datenanalyse
banking.shape
# %%
banking.head(2)

#%% 
banking.info()

#%%
banking.describe().T

#%% Welche Jobs wurden erfasst?
banking["job"].unique()

#%% Wieviele Personen gab es mit den jeweiligen Jobs?
from collections import Counter
Counter(banking["job"])

#%%
banking["y"].value_counts()

#%% Zielgröße: wie ist die Verteilung?
Counter(banking["y"])

#%% Zielgröße: wie ist die Verteilung?
import numpy as np
1 - np.sum(banking["y"]) / banking.shape[0]

#%% 
banking.loc[banking["y"]==1, "y"].count()

#%%
from plotnine import ggplot, aes, geom_bar
ggplot(banking) + aes(x="y") + geom_bar()

#%%
banking.info()

#%% One Hot Encoding für die Codierung der kategorischen (string) Merkmale / Dummy Encoding
banking_dummy = pd.get_dummies(banking, dtype=int, drop_first=True)
banking_dummy


#%% Trennung von unabh. und abh. Variablen (spaltenweise Trennung)
X = banking_dummy.drop("y", axis=1)
y = banking_dummy["y"]

#%% Train Test split (zeilenweise Trennung der Daten)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

#%% Skalierung der X_Train und X_Test-Daten
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # nur auf Trainingsdaten fitten
X_test_scaled = scaler.transform(X_test)

# %% Modellierung
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#%% Modell trainieren (auf den Trainingsdaten)
model.fit(X_train_scaled, y_train)
# %% Vorhersagen erstellen
y_test_pred = model.predict(X_test_scaled)

# %% Evaluierung der Ergebnisse
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# show axis labels directly via seaborn heatmap
plt.figure(figsize=(10, 10))    
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
#%%
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred) / len(y_test)



sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], cmap="Blues", cbar=False)
plt.show()

#%% Genauigkeit
accuracy_score(y_true=y_test, y_pred=y_test_pred)
# %% Modellvergleich
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

classifiers = {
    "KNN": KNeighborsClassifier(),
    "LinearSVC": LinearSVC(random_state=42),
    "GaussianNB": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    
    "LogisticRegression": LogisticRegression(max_iter=1000),      # linear, probabilistisch
    "DecisionTree": DecisionTreeClassifier(random_state=42),      # regelbasiert, Baum
    "SVC_RBF": SVC(kernel="rbf"),                                 # Kernel-Methode
    "MLP": MLPClassifier(max_iter=300, random_state=42),          # neuronales Netz
    "LDA": LinearDiscriminantAnalysis(),                          # Diskriminanzanalyse
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    results[name] = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"{name}: {results[name]:.4f}")

# Accuracy (Testdaten):
# KNN:              0.8958
# LinearSVC:        0.9091
# GaussianNB:       0.7472
# GradientBoosting: 0.9190
# AdaBoost:         0.9075
# LogisticRegression: 0.9128
# DecisionTree:       0.8886
# SVC_RBF:            0.9067
# MLP:                0.8946
# LDA:                0.9108

#%% Balkendiagramm der Accuracy je Modell
from plotnine import geom_col, coord_flip, labs, scale_y_continuous, geom_text, geom_col, ylim
#%%
df_results = (
    pd.DataFrame({"Modell": list(results.keys()), "Accuracy": list(results.values())})
    .sort_values("Accuracy")
)
df_results["Modell"] = pd.Categorical(df_results["Modell"], categories=df_results["Modell"], ordered=True)

(
    ggplot(df_results)
    + aes(x="Modell", y="Accuracy")
    + geom_col(fill="steelblue")
    + geom_text(aes(label="Accuracy"), format_string="{:.3f}", nudge_y=0.01, size=8)
    + coord_flip()
    + scale_y_continuous(limits=(0.0, 1))
    + labs(title="Modellvergleich: Accuracy auf Testdaten", x="", y="Accuracy")
)

# %%
