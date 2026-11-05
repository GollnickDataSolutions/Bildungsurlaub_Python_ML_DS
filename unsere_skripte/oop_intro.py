#%% packages

#%% Class definieren
class Pet:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def hello(self):
        print(f"Hello. My name is {self.name} and I am a {self.species}")


#%% eine konkrete Instanz der Klasse erstellen
kiki = Pet(name='kiki', species='dog')
# %%
kiki.hello()

kiki.species