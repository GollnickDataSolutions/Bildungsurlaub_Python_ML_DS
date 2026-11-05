#%% Pakete
import pandas as pd
from plotnine import *

#%% Datenimport
penguins = pd.read_csv("../001_Datasets/penguins.csv")
penguins.head()

#%%
# Einstieg

#%% 1. Gentoo-Pinguine sind schwerer als die anderen Arten. → Boxplot
(ggplot(penguins.dropna(subset=["body_mass_g"]))
 + aes(x="species", y="body_mass_g", fill="species")
 + geom_boxplot()
 + labs(title="Körpermasse nach Art", x="Art", y="Körpermasse (g)")
)

#%% 2. Männchen sind schwerer als Weibchen. → Boxplot
(ggplot(penguins.dropna(subset=["body_mass_g", "sex"]))
 + aes(x="sex", y="body_mass_g", fill="sex")
 + geom_boxplot()
 + labs(title="Körpermasse nach Geschlecht", x="Geschlecht", y="Körpermasse (g)")
)
#%%
penguins.groupby(["species", "island"], observed=True)["body_mass_g"].agg(["median"])

#%% 3. Auf jeder Insel leben alle drei Arten. → Gestapeltes Balkendiagramm
(ggplot(penguins.dropna(subset=["island", "species"]))
 + aes(x="island", fill="species")
 + geom_bar(position="stack")
 + labs(title="Arten je Insel", x="Insel", y="Anzahl", fill="Art")
)

#%% 4. Die Körpermasse ist ungefähr normalverteilt. → Histogramm
(ggplot(penguins.dropna(subset=["body_mass_g"]))
 + aes(x="body_mass_g")
 + geom_histogram(bins=20, fill="steelblue", color="white")
 + labs(title="Verteilung der Körpermasse", x="Körpermasse (g)", y="Anzahl")
)

#%%
# Zusammenhänge

#%% 5. Je länger die Flosse, desto schwerer der Pinguin. → Streudiagramm
(ggplot(penguins.dropna(subset=["flipper_length_mm", "body_mass_g"]))
 + aes(x="flipper_length_mm", y="body_mass_g")
 + geom_point(alpha=0.6)
 + geom_smooth(method="lm", se=False, color="red")
 + labs(title="Flossenlänge vs. Körpermasse",
        x="Flossenlänge (mm)", y="Körpermasse (g)")
)

#%% 6. Pinguine mit längerem Schnabel haben auch einen tieferen Schnabel. → Streudiagramm
(ggplot(penguins)
 + aes(x="bill_length_mm", y="bill_depth_mm", color="species")
 + geom_point(alpha=0.6) 
 + facet_wrap("species")
 + geom_smooth(method="lm", se=False)
 + labs(title="Schnabellänge vs. Schnabeltiefe",
        x="Schnabellänge (mm)", y="Schnabeltiefe (mm)")
)

#%%
# Für Fortgeschrittene

#%% 7. Die Arten lassen sich allein an den Schnabelmaßen unterscheiden. → Streudiagramm
(ggplot(penguins)
 + aes(x="bill_length_mm", y="bill_depth_mm", color="species")
 + geom_point(alpha=0.7)
 + labs(title="Schnabelmaße nach Art",
        x="Schnabellänge (mm)", y="Schnabeltiefe (mm)", color="Art")
)

#%% 8. Die Pinguine sind von 2007 bis 2009 schwerer geworden. → Liniendiagramm
# Hinweis: year fehlt in der lokalen CSV → Datensatz mit year laden
penguins_year = pd.read_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "master/inst/extdata/penguins.csv"
)
mass_by_year = (
    penguins_year.dropna(subset=["year", "body_mass_g"])
    .groupby("year", as_index=False)["body_mass_g"]
    .mean()
)
(ggplot(mass_by_year)
 + aes(x="year", y="body_mass_g")
 + geom_line()
 + geom_point(size=3)
 + labs(title="Mittlere Körpermasse pro Jahr",
        x="Jahr", y="Mittlere Körpermasse (g)")
 + scale_x_continuous(breaks=[2007, 2008, 2009])
)

#%% 9. Adelie-Pinguine sind auf allen drei Inseln gleich groß. → Boxplot
adelie = penguins[penguins["species"] == "Adelie"].dropna(
    subset=["island", "bill_length_mm"]
)
(ggplot(adelie)
 + aes(x="island", y="bill_length_mm", fill="island")
 + geom_boxplot()
 + labs(title="Schnabellänge der Adelie-Pinguine je Insel",
        x="Insel", y="Schnabellänge (mm)")
)
