# Übung: Star Wars Charaktere analysieren (Pandas Einsteiger)

Datei: `Starwars.csv` (Spalten: `height`, `mass`, `name`)

## Teil 1 – Datei laden
1. Importiere `pandas` als `pd`.
2. Lade `Starwars.csv` in ein DataFrame `df`.
3. Gib die ersten 5 Zeilen mit `.head()` aus.
4. Wie viele Zeilen und Spalten hat der Datensatz?

## Teil 2 – Explorative Datenanalyse (EDA)
5. Gib die Datentypen der Spalten aus (`.dtypes`).
6. Prüfe, ob es fehlende Werte gibt (`.isna().sum()`). Tipp: In der CSV stehen manche fehlenden Werte als `"NA"` (Text).
7. Lass dir mit `.describe()` die statistische Übersicht von `height` und `mass` anzeigen.

## Teil 3 – Wichtige Parameter extrahieren
8. Berechne die durchschnittliche Körpergröße (`height`) und Masse (`mass`).
9. Finde den größten Charakter (max. `height`) – gib Name und Größe aus.
10. Finde den schwersten Charakter (max. `mass`) – gib Name und Masse aus.
11. Wie viele Charaktere sind größer als 180 cm?
12. Bonus: Erstelle eine neue Spalte `bmi` = `mass / (height/100)**2` und finde den Charakter mit dem höchsten BMI.

## Bonus-Fragen
- Was fällt dir bei Jabba Desilijic Tiure auf (Größe/Masse)? Wie beeinflusst das den Mittelwert?
- Was passiert mit `.describe()`, wenn fehlende Werte nicht korrekt als NaN erkannt wurden?