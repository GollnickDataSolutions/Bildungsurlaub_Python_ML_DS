#%%
print("Hello World") 

#%% Datentypen
value_int = 5
type(value_int)

#%% 
value_float = 1.
type(value_float)

#%%
my_first_name = 'Bert'
my_last_name = "Gollnick"
print(my_first_name + " " + my_last_name)
print(f"{my_first_name} {my_last_name}")

#%%
my_first_name, my_last_name = "Bert", "Gollnick"


# %% Liste
vornamen = ["Garrett", "Lisa", "Dilyana"]
vornamen.pop()
vornamen.append("Antje")
vornamen

#%% Sets
s1 = set([1, 1, 2, 2, 3, 4, 5])
s1

s2 = {1, 2, 3, 4, 5}

#%% Conditional Statements
my_condition = False
if my_condition:
    print('Condition ist True')
else:
    print('Condition ist False')

#%% Algorithmus wählt zufällig eine Zahl (1-6)
# Nutzer rät eine Zahl
# Wenn richtig geraten --> Ausgabe 'Herzlichen Glückwunsch'
# Wenn falsch geraten --> Ausgabe 'Versuche es nochmal'
import random
dice_roll = random.randint(1, 6)
user_choice = input("Rate eine Zahl zwischen 1 und 6")
if dice_roll == int(user_choice):
    # == prüft Gleichheit, != prüft Ungleichheit, >, <, >=, <=
    print('Herzlichen Glückwunsch')
else:
    print('Versuche es nochmal')
# %% Schleifen
for char in 'Hamburg':
    print(char)

#%% Schleife über die Zahlen von 1-100
sum = 0
for i in range(1, 101):
    # sum = sum + i
    sum += i
    print(i)

print(f"Gesamtsumme: {sum}")

# %% Übung: Countdown
# Zahlen werden von 10 runtergezählt bis 0
# bei 0 kommt die Aussage "Takeoff"
# bei range kann man auch 3 variablen übergeben(start, stop, step)
# step -1
# sekundengenaue Darstellung mit time
import time
time.sleep(1)

#%% Dictionaries
# key: value
acronyms = {
    'VW': 'Volkswagen',
    'DB': 'Deutsche Bahn'
}
acronyms['DB'] = "Deutsche Bank"
# %%
acronyms
# %% Web-Abfrage der ISS Position
api_endpoint = "http://api.open-notify.org/iss-now.json"
import requests
response = requests.get(api_endpoint)
# %%
response_dict = response.json()
# extrahiert die iss_position und dabei dann timestamp, longitude und latitude --> timestamp, iss_long, iss_lat
timestamp = response_dict['timestamp']
iss_long = response_dict['iss_position']['longitude']
iss_lat = response_dict['iss_position']['latitude']

#%% Funktionen
def hello_world(name:str = 'Max Mustermann'):
    return f"Hello {name}"

hello_world(name="Lisa")

#%% Funktion "name_age" bekommt als Parameter: Name und Alter
# Rückgabe lautet: Name ist Alter Jahre alt!

#%% Arbeit mit Dateien
import os
current_wd =os.getcwd()  # aktuelles Arbeitsverzeichnis erfahren
#%%
# os.listdir(current_wd)
work_files_path = os.path.join(current_wd, "work_files")
os.listdir(work_files_path)