#%% Pakete
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_col, geom_hline, geom_point, geom_text, geom_boxplot,
    geom_jitter, coord_flip, coord_cartesian, scale_fill_distiller,
    scale_fill_manual, scale_color_manual, labs, theme_bw, theme,
    element_text,
)

# Farbskala analog zu RdYlGn (rot = niedrig, grün = hoch)
kategorie_farben = {"niedrig": "#d73027", "mittel": "#fee08b", "hoch": "#1a9850"}

#%% Data Import
# Der Datensatz enthält den durchschnittlichen Zufriedenheitswert (Mean, Skala 0-10)
# je Land und Geschlecht sowie die Anzahl der Befragten (N=).
file_path = "happiness.csv"
df_happiness = pd.read_csv(file_path)
df_happiness

#%% Vertrautmachen (Exploratory Data Analysis)
df_happiness.shape       # Anzahl Zeilen und Spalten
df_happiness.columns

#%% Aufräumen: Spalten umbenennen
# "N=" ist als Spaltenname unpraktisch -> "Befragte"
df_happiness = df_happiness.rename(columns={"N=": "Befragte"})
df_happiness.columns

#%% Aufräumen: fehlende Länder-Codes auffüllen
# Im CSV steht der Länder-Code nur in der ersten von drei Zeilen (Male/Female/Both).
# Mit forward-fill übernehmen wir den Code in die darunterliegenden Zeilen.
df_happiness["Country"] = df_happiness["Country"].ffill()
df_happiness.head(6)

#%% numerische Spalten besser verstehen
df_happiness.describe()

#%% Aggregat-Zeilen von echten Ländern trennen
# CC3, EU 15, NMS12 und EU 27 sind Ländergruppen, keine einzelnen Länder.
aggregat_codes = ["CC3", "EU 15", "NMS12", "EU 27"]

filter_aggregat = df_happiness["Country"].isin(aggregat_codes)
df_gruppen = df_happiness[filter_aggregat]
df_laender = df_happiness[~filter_aggregat]

df_laender.shape, df_gruppen.shape

#%% Filtern: nur Gesamtwerte (Geschlecht "Both") je Land
filter_both = df_laender["Gender"] == "Both"
df_both = df_laender[filter_both].copy()
df_both.shape

#%% Filtern: die zufriedensten Länder (Mean >= 8)
filter_glücklich = df_both["Mean"] >= 8.0
df_both[filter_glücklich].sort_values("Mean", ascending=False)

#%% neue Spalte: Zufriedenheits-Kategorie
# Wir teilen die Länder in Kategorien ein (binning).
df_both["Kategorie"] = pd.cut(
    df_both["Mean"],
    bins=[0, 6.5, 7.5, 8.5],
    labels=["niedrig", "mittel", "hoch"],
)
df_both[["Country", "Mean", "Kategorie"]].sort_values("Mean")

#%% neue Spalte: Abweichung vom EU-27-Durchschnitt
eu27_mean = df_gruppen.loc[
    (df_gruppen["Country"] == "EU 27") & (df_gruppen["Gender"] == "Both"), "Mean"
].iloc[0]

df_both["Abw_EU27"] = (df_both["Mean"] - eu27_mean).round(2)
df_both[["Country", "Mean", "Abw_EU27"]].sort_values("Abw_EU27", ascending=False)

#%% neue Spalte: Gender-Gap (Differenz Männer - Frauen) je Land
# Dazu formen wir die Tabelle um (pivot), sodass jedes Geschlecht eine Spalte wird.
df_pivot = df_laender.pivot_table(
    index="Country", columns="Gender", values="Mean"
)
df_pivot["Gender_Gap"] = (df_pivot["Male"] - df_pivot["Female"]).round(2)
df_pivot = df_pivot.sort_values("Gender_Gap", ascending=False)
df_pivot

#%% Visualisierung 1: Ranking der Länder nach Zufriedenheit
df_rank = df_both.sort_values("Mean", ascending=True).copy()
# Reihenfolge der Balken über eine geordnete Kategorie festlegen (ersetzt sort in matplotlib)
df_rank["Country"] = pd.Categorical(
    df_rank["Country"], categories=df_rank["Country"], ordered=True
)

p1 = (
    ggplot(df_rank, aes(x="Country", y="Mean", fill="Mean"))
    + geom_col()
    + geom_hline(yintercept=eu27_mean, linetype="dashed", color="black", size=1)
    + scale_fill_distiller(type="div", palette="RdYlGn", direction=1, name="Mean")
    + coord_flip()
    + labs(
        x="",
        y="Durchschnittliche Zufriedenheit (0-10)",
        title="Lebenszufriedenheit nach Land (Gesamt)",
        subtitle=f"Gestrichelte Linie: EU-27 Schnitt ({eu27_mean})",
    )
    + theme_bw()
)
p1

#%% Visualisierung 2: Gender-Gap je Land (divergierendes Balkendiagramm)
df_gap = df_pivot.dropna(subset=["Gender_Gap"]).drop(index=aggregat_codes, errors="ignore")
df_gap = df_gap.sort_values("Gender_Gap").reset_index()
# Geordnete Kategorie für die Balkenreihenfolge + Richtung für die Farbe
df_gap["Country"] = pd.Categorical(
    df_gap["Country"], categories=df_gap["Country"], ordered=True
)
df_gap["Richtung"] = np.where(
    df_gap["Gender_Gap"] < 0, "Frauen zufriedener", "Männer zufriedener"
)

p2 = (
    ggplot(df_gap, aes(x="Country", y="Gender_Gap", fill="Richtung"))
    + geom_col()
    + geom_hline(yintercept=0, color="black", size=0.8)
    + scale_fill_manual(
        values={"Frauen zufriedener": "#d73027", "Männer zufriedener": "#4575b4"},
        name="",
    )
    + coord_flip()
    + labs(
        x="",
        y="Differenz Männer − Frauen (Mean)",
        title="Gender-Gap der Lebenszufriedenheit",
    )
    + theme_bw()
)
p2

#%% Visualisierung 3: Vergleich Männer vs. Frauen (gruppiertes Balkendiagramm)
df_mf = df_laender[df_laender["Gender"].isin(["Male", "Female"])]

p3 = (
    ggplot(df_mf, aes(x="Country", y="Mean", fill="Gender"))
    + geom_col(position="dodge")
    + scale_fill_manual(values={"Male": "#4575b4", "Female": "#d73027"})
    + coord_cartesian(ylim=(5, 9))
    + labs(
        x="",
        y="Durchschnittliche Zufriedenheit",
        title="Lebenszufriedenheit nach Geschlecht und Land",
    )
    + theme_bw()
    + theme(axis_text_x=element_text(rotation=90, ha="center"))
)
p3

#%% Visualisierung 4: Stichprobengröße vs. Zufriedenheit (Streudiagramm)
p4 = (
    ggplot(df_both, aes(x="Befragte", y="Mean", color="Kategorie"))
    + geom_point(size=4)
    # Länder-Codes als Beschriftung an die Punkte (ersetzt ax.annotate)
    + geom_text(aes(label="Country"), nudge_y=0.06, size=8, show_legend=False)
    + scale_color_manual(values=kategorie_farben)
    + labs(
        x="Anzahl Befragte",
        y="Durchschnittliche Zufriedenheit",
        title="Stichprobengröße vs. Zufriedenheit je Land",
    )
    + theme_bw()
)
print(p4)

#%% Visualisierung 5: Verteilung der Zufriedenheit je Kategorie (Boxplot)
p5 = (
    ggplot(df_both, aes(x="Kategorie", y="Mean"))
    + geom_boxplot(aes(fill="Kategorie"), show_legend=False)
    + geom_jitter(width=0.15, height=0, color="black", size=2)
    + scale_fill_manual(values=kategorie_farben)
    + labs(
        x="Kategorie",
        y="Durchschnittliche Zufriedenheit",
        title="Verteilung der Länder je Zufriedenheits-Kategorie",
    )
    + theme_bw()
)
print(p5)

#%% Zusammenfassung: Kennzahlen je Kategorie
df_both.groupby("Kategorie", observed=True).agg(
    Anzahl_Laender=("Country", "count"),
    Mittelwert=("Mean", "mean"),
    Befragte_gesamt=("Befragte", "sum"),
).round(2)

# %%
