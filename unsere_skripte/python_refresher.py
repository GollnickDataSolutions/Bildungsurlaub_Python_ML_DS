#%% Datentypen: Boolean
value_bool = False #True
# %% String
my_first_name = "Bert"
my_last_name = "Gollnick"

# Stringconcatenation (Verbindung von Strings)
print(my_first_name + " " + my_last_name)
print(f"{my_first_name} {my_last_name}")
# %% Zahlentypen: Ganzzahlig (integer)
my_int = 1
type(my_int)

# %% Zahlentypen: Gleitkomma (float)
my_float = 1.001
type(my_float)

#%% Multiassignment
my_first_name, my_last_name = "Bert", "Gollnick"

#%% Conditional Statements
my_condition = True
if my_condition:
    # Body für Condition ist True
    print("Bedingung ist wahr")
else:
    # Body für Condition ist Falsch
    print("Bedingung ist falsch")

#%% Würfelwurf
import random
wuerfelwurf = random.randint(1, 6)
# wuerfelwurf = random.randint(a=1, b=6)  # alternativ
our_guess = input("Rate die gewürfelte Zahl: ")
if int(our_guess) == wuerfelwurf:
    print("Herzlichen Glückwunsch! Du hast richtig geraten")
else:
    print("Tut mir leid. Versuche es nochmal.")

#%% Vergleichsoperatoren
# == prüft auf Gleichheit
# != prüft auf Ungleichheit
# <, >, <=, >=

# %% Listen erstellen (start mit leerer Liste, dann append)
participants = []
participants.append("Patrick")
participants.append("Stefanie")
participants
#%% Liste gleich mit Werten erstellen
participants = ["Patrick", "Stefanie"]
participants.remove("Stefanie")
#%%
participants
# %%
participants.extend(["Hendrik", "Nuwar"])
# participants = participants + ["Hendrik", "Nuwar"] # alternative
participants
# %% Ist "Sophie" Teil der Participants
if "Sophie" in participants:
    print("ja, Sophie ist dabei")
else:
    print("Nein, ist sie nicht")

#%% Sets
my_set = set([1, 2, 3, 3, 4, 4, 5, 6])
my_set

#%% Gibt es Duplikate innerhalb 'participants'
if len(participants) == len(set(participants)):
    print("Es gibt keine Dopplungen.")

#%% Dictionaries
acronyms = {
    "BU": "Bildungsurlaub",
    "DS": "Data Science",
    "ML": "Machine Learning"
}
# Zugriff auf Schlüssel mit ["Schlüsselname"]
acronyms["BU"]

# Sicherer Zugriff, auch wenn der Schlüssel nicht existiert
acronyms.get("BU")

#%% REST-API Zugriff auf die ISS Position
api_endpoint = "http://api.open-notify.org/iss-now.json"
import requests
response = requests.get(api_endpoint).json()
#%%
response['iss_position']['longitude']

#%% Dateisysteminteraktion
import os
os.listdir(".")

#%% Unterordner erstellen
sub_folder = "data"
if not os.path.exists(sub_folder):
    os.makedirs(sub_folder)


#%% Dataframes
import pandas as pd
my_dict = {
    "name": ["Stuart", "Bob", "Kevin"],
    "grades": [2, 3, 1]
}
df_minions = pd.DataFrame(my_dict)

# %% Spalten und Zeilen des Dataframes ermitteln
print(f"Anzahl Zeilen: {len(df_minions)}")  # Anzahl Zeilen
print(df_minions.count(axis=1))
print(f"Shape: {df_minions.shape}") # liefert Anzahl Zeilen, Anzahl Spalten
