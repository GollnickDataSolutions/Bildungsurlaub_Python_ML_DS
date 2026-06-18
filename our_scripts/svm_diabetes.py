#%% Pakete
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
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

#%% Modellstudie 
results = []

models = [DummyClassifier(), LogisticRegression(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), HistGradientBoostingClassifier()]
for model in models:
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    results.append({"Model": model.__class__.__name__, "Accuracy": acc})
    print(f"{model.__class__.__name__}: {acc}")
    print("-" * 20)

df_results = pd.DataFrame(results)
print(df_results)

#%%
from plotnine import ggplot, aes, geom_bar, ylab, xlab, ggtitle, theme, element_text

# Convert results DataFrame to suitable format for plotnine
# In case Model is index
df_results_reset = df_results.reset_index() if df_results.index.name == 'Model' else df_results

plot = (
    ggplot(df_results_reset, aes(x='Model', y='Accuracy', fill='Model'))
    + geom_bar(stat='identity', show_legend=False)
    + ylab('Accuracy')
    + xlab('Model')
    + ggtitle('Modellvergleich auf Testdaten')
    + theme(axis_text_x=element_text(rotation=45, hjust=1))
)
plot.show()



#%% 
sns.heatmap(confusion_matrix(y_true=y_test, y_pred=y_test_pred), annot=True)
plt.show()


# %%
