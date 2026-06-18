#%% Pakete
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%% Data import
diabetes = pd.read_csv("diabetes.csv")
diabetes

#%% X, y extrahieren
X, y = diabetes.drop(columns=["Outcome"]), diabetes["Outcome"]
#%% training und test daten erstellen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% modeling
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
# model.coef_

#%% Evaluierung
y_test_pred = model.predict(X_test_scaled)
accuracy_score(y_true=y_test, y_pred=y_test_pred)

#%% Dummy Classifier
import numpy as np
# 1-np.sum(y_test) / len(y_test)
dummy_clf = DummyClassifier()
dummy_clf.fit(X=X_train_scaled, y=y_train)
dummy_clf.score(X=X_test_scaled, y=y_test)

#%% 
sns.heatmap(confusion_matrix(y_true=y_test, y_pred=y_test_pred), annot=True)
plt.show()