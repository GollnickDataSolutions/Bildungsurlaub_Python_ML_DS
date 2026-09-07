#%%
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
data = fetch_openml('adult', version=2, as_frame=True)
# %%
data
# %%
data.keys()
# %%
X, y =  data["data"], data["target"].isin(["<=50K"])
X.drop(columns=["workclass", "occupation", "native-country"], inplace=True)

# %%
y = [0 if i==True else 1 for i in y]
y
# %% training und test daten erstellen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% data scaling
scaler = StandardScaler()  # StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled

# %%
results = []

models = [DummyClassifier(), LogisticRegression(), SVC(), RandomForestClassifier(), GradientBoostingClassifier()] 
for model in models:
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    results.append({"Model" : model.__class__.__name__, "Accuracy" : acc})
    print(f"{model.__class__.__name__} : {acc}")
    print("♥"*20)

df_results = pd.DataFrame(results)
df_results

# %%
results = []

models = [DummyClassifier(), LogisticRegression(), SVC(), RandomForestClassifier(), GradientBoostingClassifier()] 
for model in models:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    results.append({"Model" : model.__class__.__name__, "Accuracy" : acc})
    print(f"{model.__class__.__name__} : {acc}")
    print("♥"*20)

df_results = pd.DataFrame(results)
df_results

# %% modeling
model = SVC()
model.fit(X_train_scaled, y_train)
#model.coef_

#%% Testen des Modells
y_test_pred = model.predict(X_test_scaled)
#y_test_pred


#%% Bestimmtheitsmaß (R^2) ermitteln
accuracy_score(y_true=y_test, y_pred=y_test_pred)

# %%
dummy_clf = DummyClassifier()
dummy_clf.fit(X=X_train_scaled, y=y_train)
dummy_clf.score(X=X_test_scaled, y=y_test)

# %%
sns.heatmap(confusion_matrix(y_true=y_test, y_pred=y_test_pred), annot=True)