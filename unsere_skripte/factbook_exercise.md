# Übungen: EDA mit pandas

## Über den Datensatz

factbook.csv enthält Kennzahlen aus dem CIA World Factbook für 263 Länder und Gebiete in 45 Spalten.

[Factbook Dataset](https://www.kaggle.com/datasets/lucafrance/the-world-factbook-by-cia)

Zwei Stolperfallen solltet ihr selbst entdecken: Die Datei ist nicht mit Kommas getrennt, und die zweite Zeile enthält keine echten Daten. Schaut euch die Rohdatei deshalb zuerst in einem Texteditor an.

## Teil 1: Daten laden und erster Überblick

1. Lest `factbook.csv` mit `pd.read_csv()` in einen DataFrame `df` ein. Prüft danach mit `df.head()`, ob die Spalten korrekt getrennt sind. Falls nicht, findet den passenden Parameter.
2. Schaut euch mit `df.dtypes` die Datentypen an. Warum sind fast alle Spalten Text statt Zahl? Passt das Einlesen so an, dass die Zeile mit den Typangaben (`String`, `double`, …) übersprungen wird.
3. Wie viele Zeilen und Spalten hat der Datensatz jetzt?
4. Lasst euch mit `df.info()` eine Übersicht ausgeben. Welche Spalte hat die meisten Nicht-Null-Werte, welche die wenigsten?
5. Zeigt 5 zufällige Länder mit `df.sample()` an.
6. Erstellt mit `df.describe()` eine Statistik der numerischen Spalten. Tipp: `.T` macht die Ausgabe lesbarer.

## Teil 2: Fehlende Werte und Datenqualität

1. Zählt die fehlenden Werte pro Spalte mit `isna().sum()`.
2. Welche 5 Spalten sind am lückenhaftesten?
3. Gibt es doppelte Länder? Prüft das mit `duplicated()`.
4. Wie viele eindeutige Werte hat jede Spalte? Nutzt `nunique()`.
5. Seht euch die Spalte `Country` genauer an. Gibt es Einträge, die gar keine Länder sind? Tipp: Sucht nach Namen, die mit `#` beginnen.
6. Erstellt einen bereinigten DataFrame `laender`, der nur echte Länder enthält. Mit diesem DataFrame arbeitet ihr ab jetzt weiter.

## Teil 3: Filtern von Daten

1. Zeigt alle Daten zu Deutschland an.
2. Welche Länder haben mehr als 100 Millionen Einwohner? Sortiert sie absteigend nach `Population`.
3. In welchen Ländern liegt die Lebenserwartung (`Life expectancy at birth(years)`) über 80 Jahren?
4. Welche Länder haben eine Arbeitslosenquote (`Unemployment rate(%)`) von mehr als 30 %?
5. Welche Länder haben mehr als 50 Millionen Einwohner **und** ein BIP pro Kopf (`GDP - per capita`) unter 2.000?
6. Welche Länder haben eine Geburtenrate über 40 **oder** eine Kindersterblichkeit (`Infant mortality rate(deaths/1000 live births)`) über 100?
7. Findet alle Länder, die kein Öl fördern (`Oil - production(bbl/day)` gleich 0), aber mehr als 500.000 Barrel pro Tag verbrauchen.
8. Filtert die Länder Deutschland, Frankreich, Italien, Spanien und Polen mit `isin()`. Achtung: Die Ländernamen sind auf Englisch. Zeigt nur `Country`, `Population` und `GDP - per capita`.
