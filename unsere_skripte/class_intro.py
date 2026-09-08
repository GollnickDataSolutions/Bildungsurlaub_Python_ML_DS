#%% Pakete
import random

#%% Klasse für Kartenspiel
class Deck:
    """ Class for 32 cards deck
    """
    SUIT = ["♣", "♠", "♦", "♥"]
    RANK = [str(i) for i in range(7, 11)] + ["J", "Q", "K", "A"]

    def __init__(self):
        self.cards = []
        for suit in self.SUIT:
            for rank in self.RANK:
                self.cards.append(f"{suit} {rank}")
    
    def shuffle(self):
        random.shuffle(self.cards)
        print(self.cards)
        return "Die Karten wurden erfolgreich gemischt."
    
    def draw(self):
        return self.cards.pop()

my_deck = Deck()

#%%
my_deck.shuffle()
my_deck.cards

#%%
my_deck.draw()
#%%
my_deck.cards

#%% Exkurs
zahlen_liste = []
for i in range(7,11):
    zahlen_liste.append(str(i))
zahlen_liste

#%%
[str(i) for i in range(7, 11)]