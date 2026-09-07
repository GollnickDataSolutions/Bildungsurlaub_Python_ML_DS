#%% Pakete
import pandas as pd

#%% Datentypen
# int
value_int = 1
type(value_int)
# double / float
val_float = 1.5
type(val_float)

# bool
val_bool = True  # oder: False
type(val_bool)

# string
my_first_name = "Bert"
# my_first_name = "Bert's Kurs"
# my_first_name = 'Bert\'s Kurs'
my_last_name = "Gollnick"

# String concatenation
my_first_name + " " + my_last_name
# Alternative: f-string
f"{my_first_name} {my_last_name}, {value_int}"

#%% conditional statements (if then else)
my_condition = True
if my_condition:
    print("Die Bedingung ist wahr")
else:
    print("Die Bedingung ist NICHT wahr")


# %% Würfel-Simulation
# import random
import numpy as np

wurf_ergebnis = np.random.randint(1, 7)
nutzer_vorhersage = int(input("Vorhersage: "))
if wurf_ergebnis == nutzer_vorhersage:
    print("richtig geraten")
else:
    print("Da hast du falsch gelegen.")


#%%
import numpy as np
Temperatur = np.random.randint(0, 36)
es_regnet = True
if Temperatur >= 35:
    print ("Bleibe im Kühlen")
elif Temperatur <= 35 and es_regnet:
    print ("Nimm einen Schirm mit!")
# %%
import numpy as pd
Temperatur = np.random.randint(0, 36)
es_regnet = True
if Temperatur >= 35:
    print ("Bleibe im Kühlen")
elif Temperatur <= 35 and es_regnet :
    print ("Nimm einen Schirm mit!")
 
# %% Listen
teilnehmer = []
teilnehmer.append('Jana')
teilnehmer.append('Elena')
teilnehmer.append('Manfred')
#%%
teilnehmer.remove('Elena')
#%%
teilnehmer

#%%
teilnehmer[1]

#%% prüfe, ob Jana Teilnehmer ist
# wenn ja, Ausgabe 'Hallo Jana, willkommen im Kurs'
if 'Jana' in teilnehmer:
    print(f"Hallo Jana, willkommen im Kurs")

#%% Sets
teilnehmer = set(teilnehmer)
teilnehmer = {'Jana', 'Manfred', 'Elena', 'Jana'}
teilnehmer
# %%
wochentage = {'mo', 'di', 'mi'}

#%% Dictionary
acronyme = {
    "LIFE": "Learning is fun and exciting",
    "KISS": "Keep it simple stupid",
    "TEAM": "Toll ein Anderer machts",
    "Thema1": {"tbd": "muss noch definiert werden"}
}
acronyme["Thema1"]["tbd"] = "weiß ich immer noch nicht"
acronyme

#%%
import requests
API_ENDPOINT = "http://api.open-notify.org/iss-now.json"
response = requests.get(url=API_ENDPOINT)


#%%
response_json = response.json()
response_json["iss_position"]["latitude"]


#%% Schleifen
# countdown
# Ausgabe: 10
# Ausgabe: 9
# Ausgabe: ...
# Ausgabe (0): 'Frohes Neues Jahr'


for i in range(1, 10):
    print(i)
# %% Ausgaben summieren
ausgaben = [1, 5.5, 8, 25, 3]
sum = 0

#%% Funktionen
def hello_world():
    return "Hello world."

hello_world()


# %% Funktion mit einem Parameter
def hello_person(name:str = "Max Mustermann") -> str:
    """ Begrüßt freundlich einen Nutzer mit seinem Namen.

    Args:
        name:  der Name des Nutzers, der begrüßt wird
    Returns:
        eine freundliche Begrüßung des Nutzers
    """
    return f"Hello {name}"

hello_person(name = "Tina")

# %% Caesar Chiffre (Verschlüsselung)
#  Modulo Operator gibt uns den Rest der Division zurück: 28 % 26
