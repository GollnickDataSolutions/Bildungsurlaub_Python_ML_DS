#%% Pakete
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================
# Teil 1: Daten laden und erster Überblick
# =============================================================
#%% 1.1 Einlesen mit falschem Trennzeichen (Stolperfalle 1)
factbook = Path(__file__).parent / "factbook.csv"
df = pd.read_csv(factbook)
print(df.head())  # eine einzige Spalte -> Trennzeichen ist ';'

#%% 1.2 Trennzeichen + Typzeile überspringen (Stolperfalle 2)
df = pd.read_csv(factbook, sep=";")
print(df.dtypes)  # fast alles object, weil Zeile 2 'String'/'double' enthält
df = pd.read_csv(factbook, sep=";", skiprows=[1])
print(df.dtypes)

#%% 1.3 Zeilen und Spalten
print(df.shape)

#%% 1.4 info()
df.info()
nn = df.notna().sum()
print("meiste Nicht-Null:", nn.idxmax(), nn.max())
print("wenigste Nicht-Null:", nn.idxmin(), nn.min())

#%% 1.5 Zufällige Länder
print(df.sample(5, random_state=42))

#%% 1.6 describe
print(df.describe().T)

#%% 1.7 Größte / kleinste Fläche
area = df.dropna(subset=["Area(sq km)"])
print("Größte:", area.loc[area["Area(sq km)"].idxmax(), ["Country", "Area(sq km)"]].tolist())
print("Kleinste:", area.loc[area["Area(sq km)"].idxmin(), ["Country", "Area(sq km)"]].tolist())
print(area.sort_values("Area(sq km)")[["Country", "Area(sq km)"]].head(3))
print(area.sort_values("Area(sq km)")[["Country", "Area(sq km)"]].tail(3))

# =============================================================
# Teil 2: Fehlende Werte und Datenqualität
# =============================================================
#%% 2.1 Fehlende Werte pro Spalte
print(df.isna().sum())

#%% 2.2 Anteil in Prozent, Top 5
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing_pct)
print("Lückenhafteste 5:\n", missing_pct.head(5))

#%% 2.3 Doppelte Länder
print("Doppelte Zeilen:", df.duplicated().sum())
print("Doppelte Länder:", df["Country"].duplicated().sum())
print(df[df["Country"].duplicated(keep=False)])

#%% 2.4 Eindeutige Werte
print(df.nunique())

#%% 2.5 Einträge in Country, die keine Länder sind
print(df[df["Country"].str.startswith("#")]["Country"])

#%% 2.6 Bereinigter DataFrame
laender = df[~df["Country"].str.startswith("#")].copy()
print(laender.shape)

# =============================================================
# Teil 3: Filtern von Daten
# =============================================================
# --- Einfache Filter ---
#%% 3.1 Deutschland
print(laender[laender["Country"] == "Germany"].T)

#%% 3.2 Mehr als 100 Mio. Einwohner
print(
    laender[laender["Population"] > 100_000_000]
    .sort_values("Population", ascending=False)[["Country", "Population"]]
)

#%% 3.3 Lebenserwartung > 80
print(laender[laender["Life expectancy at birth(years)"] > 80][
    ["Country", "Life expectancy at birth(years)"]])

#%% 3.4 Arbeitslosenquote > 30 %
print(laender[laender["Unemployment rate(%)"] > 30][["Country", "Unemployment rate(%)"]])

# --- Kombinierte Filter ---
#%% 3.5 > 50 Mio. Einwohner UND BIP pro Kopf < 2000
print(laender[(laender["Population"] > 50_000_000) & (laender["GDP - per capita"] < 2000)][
    ["Country", "Population", "GDP - per capita"]])

#%% 3.6 Geburtenrate > 40 ODER Kindersterblichkeit > 100
mask = (laender["Birth rate(births/1000 population)"] > 40) | (
    laender["Infant mortality rate(deaths/1000 live births)"] > 100
)
print(laender[mask][["Country", "Birth rate(births/1000 population)",
                     "Infant mortality rate(deaths/1000 live births)"]])

#%% 3.7 Kein Öl gefördert, aber > 500.000 bbl/Tag verbraucht
print(laender[(laender["Oil - production(bbl/day)"] == 0)
              & (laender["Oil - consumption(bbl/day)"] > 500_000)][
    ["Country", "Oil - production(bbl/day)", "Oil - consumption(bbl/day)"]])

# --- Weitere Filtermethoden ---
#%% 3.8 isin()
europa = ["Germany", "France", "Italy", "Spain", "Poland"]
print(laender[laender["Country"].isin(europa)][["Country", "Population", "GDP - per capita"]])

#%% 3.9 str.contains()
print(laender[laender["Country"].str.contains("Island", case=False, na=False)]["Country"])

#%% 3.10 between()
print(laender[laender["Life expectancy at birth(years)"].between(50, 60)][
    ["Country", "Life expectancy at birth(years)"]])

#%% 3.11 query() (Aufgabe 5)
print(laender.query("Population > 50_000_000 and `GDP - per capita` < 2000")[
    ["Country", "Population", "GDP - per capita"]])

#%% 3.12 Population vorhanden, GDP fehlt
print(laender[laender["Population"].notna() & laender["GDP"].isna()][
    ["Country", "Population", "GDP"]])

# --- Top-N ---
#%% 3.13 Top 10 Internetnutzung
print(laender.nlargest(10, "Internet users")[["Country", "Internet users"]])

#%% 3.14 10 niedrigste Lebenserwartung
print(laender.nsmallest(10, "Life expectancy at birth(years)")[
    ["Country", "Life expectancy at birth(years)"]])

# =============================================================
# Teil 4: Gruppieren, Korrelationen, Verteilungen
# =============================================================
#%% 4.1 Spalte Wohlstand
laender["Wohlstand"] = pd.cut(
    laender["GDP - per capita"],
    bins=[-float("inf"), 2000, 15000, float("inf")],
    labels=["niedrig", "mittel", "hoch"],
    right=False,  # niedrig: <2000, mittel: 2000-<15000, hoch: >=15000
)
# "über 15.000" -> exakt 15.000 gehört streng genommen zu "mittel"
laender.loc[laender["GDP - per capita"] == 15000, "Wohlstand"] = "mittel"

#%% 4.2 Anzahl je Gruppe
print(laender["Wohlstand"].value_counts(dropna=False))

#%% 4.3 Lebenserwartung je Gruppe
LE = "Life expectancy at birth(years)"
print(laender.groupby("Wohlstand", observed=True)[LE].agg(["mean", "median", "count"]))

#%% 4.4 Geburtenrate und Kindersterblichkeit je Gruppe
BR = "Birth rate(births/1000 population)"
IM = "Infant mortality rate(deaths/1000 live births)"
print(laender.groupby("Wohlstand", observed=True)[[BR, IM]].mean())
# Auffall: Je ärmer die Gruppe, desto höher Geburtenrate und Kindersterblichkeit.

#%% 4.5 Korrelationsmatrix
FERT = "Total fertility rate(children born/woman)"
cols = ["GDP - per capita", LE, BR, IM, FERT]
corr = laender[cols].corr()
print(corr)
pairs = corr.stack()
pairs = pairs[[a < b for a, b in pairs.index]]  # Diagonale und Duplikate entfernen
top = pairs.abs().idxmax()
print("Stärkster Zusammenhang:", top, round(pairs[top], 3))

#%% 4.6 Quantile
print(laender["GDP - per capita"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

#%% 4.7 Histogramm Lebenserwartung
laender[LE].plot.hist(bins=30, edgecolor="black", title="Lebenserwartung")
plt.xlabel("Jahre")
plt.show()

#%% 4.8 Boxplot Lebenserwartung je Wohlstandsgruppe
laender.boxplot(column=LE, by="Wohlstand")
plt.suptitle("")
plt.title("Lebenserwartung je Wohlstandsgruppe")
plt.show()

#%% 4.9 Schiefe
print(laender.select_dtypes("number").skew().sort_values(key=abs, ascending=False))

# =============================================================
# Bonus
# =============================================================
#%% B1 Internet pro Kopf
laender["Internet pro Kopf"] = laender["Internet users"] / laender["Population"]
print(laender.nlargest(10, "Internet pro Kopf")[["Country", "Internet pro Kopf"]])

#%% B2 Bevölkerungsdichte (> 1 Mio. Einwohner)
laender["Dichte"] = laender["Population"] / laender["Area(sq km)"]
print(laender[laender["Population"] > 1_000_000].nlargest(10, "Dichte")[
    ["Country", "Population", "Area(sq km)", "Dichte"]])

#%% B3 Exportüberschuss
laender["Handelsbilanz"] = laender["Exports"] - laender["Imports"]
print(laender.nlargest(10, "Handelsbilanz")[["Country", "Exports", "Imports", "Handelsbilanz"]])

#%% B4 quick_eda
def quick_eda(data: pd.DataFrame) -> pd.DataFrame:
    """Übersichtstabelle: Datentyp und Anteil fehlender Werte je Spalte.
    Form und Duplikate stehen in .attrs."""
    out = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing_%": (data.isna().mean() * 100).round(1),
    })
    out.attrs["shape"] = data.shape
    out.attrs["duplicates"] = int(data.duplicated().sum())
    print(f"Form: {data.shape}, Duplikate: {out.attrs['duplicates']}")
    return out


print(quick_eda(laender))

#%% B5 Eigene Frage: Haben Länder mit hoher Militärquote eine niedrigere Lebenserwartung?
MIL = "Military expenditures - percent of GDP(%)"
print(laender[[MIL, LE]].corr())
# Erkenntnis: Die Korrelation ist mit ca. 0,03 praktisch null. Hohe Militärausgaben
# (% vom BIP) sagen in diesem Datensatz nichts über die Lebenserwartung aus.
