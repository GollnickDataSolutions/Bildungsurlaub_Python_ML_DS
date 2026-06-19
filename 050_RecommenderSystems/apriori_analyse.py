"""
Apriori-Analyse des OnlineRetail-Datensatzes
=============================================

Führt eine Marktkorbanalyse (Association Rule Mining) mit dem
Apriori-Algorithmus durch. Identifiziert häufige Produktkombinationen
und generiert Assoziationsregeln (z. B. "Käufer von X kaufen auch Y").

Verwendete Metriken:
  - Support:  Häufigkeit eines Itemsets relativ zu allen Transaktionen
  - Confidence:  Bedingte Wahrscheinlichkeit P(Y | X)
  - Lift:  Wie viel häufiger wird Y mit X gekauft als erwartet (Lift > 1 = positive Korrelation)
  - Leverage:  Differenz zwischen beobachteter und erwarteter gemeinsamer Häufigkeit
  - Conviction:  Wie oft würde die Regel "falsch" vorhersagen, wenn X und Y unabhängig wären

Datenquelle: UCI Online Retail Dataset (UK-basierter Online-Shop)
"""

# ---------------------------------------------------------------------------
# 0. Importe
# ---------------------------------------------------------------------------
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

# Stelle sicher, dass Unicode in der Konsole dargestellt werden kann
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Konfigurierbare Parameter (hier exponiert zum leichten Anpassen)
# ---------------------------------------------------------------------------

# Dateipfad zum Datensatz
DATA_PATH = os.path.join("..", "001_Datasets", "OnlineRetail.xlsx")

# Land-Filter: Nur Transaktionen aus diesem Land (None = alle Länder)
COUNTRY_FILTER = "United Kingdom"

# Apriori-Parameter
MIN_SUPPORT = 0.02        # Mindest-Support für Frequent Itemsets
MIN_CONFIDENCE = 0.4      # Mindest-Confidence für Regeln
MIN_LIFT = 1.2            # Mindest-Lift für Regeln (nur positive Assoziationen)

# Metrik für das Regel-Ranking (support, confidence, lift, leverage, conviction)
RANKING_METRIC = "lift"

# Ausgabe-Ordner für Diagramme
OUTPUT_DIR = "apriori_output"

# Maximale Anzahl Itemsets/Regeln in Konsolenausgabe
TOP_N_ITEMS = 20
TOP_N_RULES = 15

# ---------------------------------------------------------------------------
# 2. Daten laden
# ---------------------------------------------------------------------------

print("=" * 70)
print("  APRIORI-MARKTKORBANALYSE — Online Retail Dataset")
print("=" * 70)

print(f"\n[1/8] Lade Daten: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)
print(f"       Zeilen: {df.shape[0]:,},  Spalten: {df.shape[1]}")
print(f"       Spalten: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3. Datenbereinigung
# ---------------------------------------------------------------------------

print(f"\n[2/8] Bereinige Daten")

# 3a. Stornierungen entfernen (InvoiceNo beginnt mit "C")
n_before = len(df)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
n_cancelled = n_before - len(df)
print(f"       Entfernte Stornierungen: {n_cancelled:,}")

# 3b. Leere CustomerID entfernen
n_before = len(df)
df = df.dropna(subset=["CustomerID"])
n_no_cust = n_before - len(df)
print(f"       Entfernte Zeilen ohne Kunden-ID: {n_no_cust:,}")

# 3c. Negative oder Null-Mengen entfernen
n_before = len(df)
df = df[df["Quantity"] > 0]
n_neg_qty = n_before - len(df)
print(f"       Entfernte Zeilen mit Quantity <= 0: {n_neg_qty:,}")

# 3d. Negative oder Null-Preise entfernen
n_before = len(df)
df = df[df["UnitPrice"] > 0]
n_neg_price = n_before - len(df)
print(f"       Entfernte Zeilen mit UnitPrice <= 0: {n_neg_price:,}")

# 3e. Fehlende Beschreibungen entfernen
n_before = len(df)
df = df.dropna(subset=["Description"])
n_no_desc = n_before - len(df)
print(f"       Entfernte Zeilen ohne Beschreibung: {n_no_desc:,}")

print(f"       → Verbleibende Zeilen: {df.shape[0]:,}")

# ---------------------------------------------------------------------------
# 4. Filtern auf das gewünschte Land
# ---------------------------------------------------------------------------

print(f"\n[3/8] Filtere auf Land: '{COUNTRY_FILTER}'")

# Zeige Verteilung der Top-Länder vor dem Filter (für Kontext)
if COUNTRY_FILTER is not None:
    country_counts = df["Country"].value_counts()
    print(f"       Top-5 Länder (vor Filter):")
    for country, count in country_counts.head(5).items():
        pct = count / len(df) * 100
        print(f"         {country:25s}  {count:8,d}  ({pct:5.2f}%)")

    df = df[df["Country"] == COUNTRY_FILTER]
    print(f"       → Verbleibende Zeilen nach Filter: {df.shape[0]:,}")

# ---------------------------------------------------------------------------
# 5. Explorative Analyse der Top-Artikel
# ---------------------------------------------------------------------------

print(f"\n[4/8] Explorative Analyse")

# Top-Artikel (nach Anzahl der Verkäufe)
top_items = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_N_ITEMS)
)
print(f"\n       Top-{TOP_N_ITEMS} Artikel (nach verkaufter Menge):")
for item, qty in top_items.items():
    print(f"         {item:45s}  {qty:>8,d}")

# Transaktionsanzahl (unique InvoiceNo)
n_transactions = df["InvoiceNo"].nunique()
print(f"\n       Gesamtanzahl Transaktionen (Rechnungen): {n_transactions:,}")
print(f"       Einzigartige Produkte: {df['StockCode'].nunique():,}")

# ---------------------------------------------------------------------------
# 6. Datenaufbereitung für den Apriori-Algorithmus
# ---------------------------------------------------------------------------

print(f"\n[5/8] Bereite Transaktionsdaten für Apriori auf")

# Gruppiere pro Rechnung alle gekauften Produkte als Liste
# Wichtig: Entferne Dubletten (gleiches Produkt 2x in einer Rechnung)
transactions = (
    df.groupby("InvoiceNo")["Description"]
    .apply(lambda x: list(set(x)))
    .tolist()
)

print(f"       Anzahl Transaktionen (Baskets): {len(transactions):,}")

# ---------------------------------------------------------------------------
# 7. Apriori-Algorithmus — Frequent Itemsets
# ---------------------------------------------------------------------------

print(f"\n[6/8] Apriori: Frequent Itemsets (min_support = {MIN_SUPPORT})")

# TransactionEncoder wandelt die Transaktionsliste in eine
# One-Hot-kodierte Matrix um (Zeilen = Transaktionen, Spalten = Produkte)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_basket = pd.DataFrame(te_ary, columns=te.columns_)

print(f"       Matrix-Dimensionen: {df_basket.shape[0]} Transaktionen × {df_basket.shape[1]} Produkte")

# Apriori-Algorithmus anwenden
frequent_itemsets = apriori(
    df_basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=None,
)

print(f"       Gefundene häufige Itemsets: {len(frequent_itemsets)}")

if len(frequent_itemsets) == 0:
    print("\n⚠️  Keine häufigen Itemsets gefunden. Versuche niedrigeren min_support.")
    print("   Setze MIN_SUPPORT auf 0.02 für diesen Durchlauf.")
    MIN_SUPPORT = 0.02
    frequent_itemsets = apriori(
        df_basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=None,
    )
    print(f"       Gefundene häufige Itemsets: {len(frequent_itemsets)}")

# Top-Itemsets nach Support ausgeben
top_itemsets = frequent_itemsets.sort_values("support", ascending=False).head(TOP_N_ITEMS)
print(f"\n       Top-{TOP_N_ITEMS} häufige Itemsets:")
for _, row in top_itemsets.iterrows():
    items = ", ".join(sorted(row["itemsets"]))
    print(f"         [{row['support']:.4f}]  {{{items}}}")

# ---------------------------------------------------------------------------
# 8. Assoziationsregeln generieren
# ---------------------------------------------------------------------------

print(f"\n[7/8] Generiere Assoziationsregeln")
print(f"       min_confidence = {MIN_CONFIDENCE},  min_lift = {MIN_LIFT}")

# Erzeuge alle Regeln aus den Frequent Itemsets
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE,
)

# Filtere nach Mindest-Lift
rules = rules[rules["lift"] >= MIN_LIFT].copy()

print(f"       Gefundene Regeln: {len(rules)}")

if len(rules) > 0:
    # Sortiere nach der gewählten Ranking-Metrik
    if RANKING_METRIC in rules.columns:
        rules_sorted = rules.sort_values(RANKING_METRIC, ascending=False)

    print(f"\n       Top-{TOP_N_RULES} Regeln (sortiert nach {RANKING_METRIC}):")
    for i, (_, row) in enumerate(rules_sorted.head(TOP_N_RULES).iterrows()):
        antecedents = ", ".join(sorted(row["antecedents"]))
        consequents = ", ".join(sorted(row["consequents"]))
        print(f"\n       Regel #{i+1}:")
        print(f"         {{{antecedents}}} → {{{consequents}}}")
        print(f"         Support: {row['support']:.4f}  "
              f"Confidence: {row['confidence']:.4f}  "
              f"Lift: {row['lift']:.4f}")

# ---------------------------------------------------------------------------
# 9. Visualisierungen
# ---------------------------------------------------------------------------

print(f"\n[8/8] Erstelle Visualisierungen")

# Lege Ausgabe-Ordner an
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definiere ein einheitliches Farbschema
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

# 9a. Balkendiagramm: Top-Artikel nach Verkaufsmenge
fig, ax = plt.subplots(figsize=(12, 7))
top_items_sorted = top_items.sort_values()
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_items_sorted)))
bars = ax.barh(range(len(top_items_sorted)), top_items_sorted.values, color=colors)
ax.set_yticks(range(len(top_items_sorted)))
ax.set_yticklabels(top_items_sorted.index, fontsize=9)
ax.set_xlabel("Verkaufte Menge", fontsize=12)
ax.set_title(f"Top-{TOP_N_ITEMS} Produkte nach Verkaufsmenge", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
sns.despine(left=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_top_produkte.png"), dpi=150)
plt.close()
print(f"       ✓ 01_top_produkte.png")

# 9b. Heatmap: Co-Occurrence der Top-Produkte
if len(frequent_itemsets) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Erstelle eine Co-Occurrence-Matrix für die TOP-Produkte
    top_product_names = list(top_items.index[:20])

    # Nur Produkte, die auch in der Basket-Matrix vorkommen
    top_product_names = [p for p in top_product_names if p in df_basket.columns]

    if len(top_product_names) >= 2:
        # Berechne wie oft zwei Produkte gemeinsam gekauft werden
        co_occurrence = np.zeros((len(top_product_names), len(top_product_names)))
        for i, p1 in enumerate(top_product_names):
            for j, p2 in enumerate(top_product_names):
                if i <= j:  # obere Dreiecksmatrix
                    both = (df_basket[p1] & df_basket[p2]).sum()
                    only_p1 = df_basket[p1].sum()
                    only_p2 = df_basket[p2].sum()
                    jaccard = both / (only_p1 + only_p2 - both) if (only_p1 + only_p2 - both) > 0 else 0
                    co_occurrence[i, j] = jaccard
                    co_occurrence[j, i] = jaccard

        mask = np.triu(np.ones_like(co_occurrence, dtype=bool), k=1)

        sns.heatmap(
            co_occurrence,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=[s[:25] for s in top_product_names],
            yticklabels=[s[:25] for s in top_product_names],
            square=True,
            cbar_kws={"label": "Jaccard-Ähnlichkeit"},
            ax=ax,
        )
        ax.set_title("Co-Occurrence der Top-Produkte (Jaccard-Ähnlichkeit)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_co_occurrence_heatmap.png"), dpi=150)
        plt.close()
        print(f"       ✓ 02_co_occurrence_heatmap.png")

# 9c. Scatter-Plot: Regel-Landschaft (Support × Confidence × Lift)
if len(rules) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        rules["support"],
        rules["confidence"],
        s=rules["lift"] * 30,  # Punktgröße proportional zum Lift
        c=rules["lift"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Lift")
    ax.set_xlabel("Support", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title("Assoziationsregeln: Support vs. Confidence (Größe = Lift)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_regel_landschaft.png"), dpi=150)
    plt.close()
    print(f"       ✓ 03_regel_landschaft.png")

# 9d. Top-Regeln als horizontale Balken
if len(rules) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    top_rules_plot = rules_sorted.head(TOP_N_RULES).copy()
    top_rules_plot["rule_label"] = top_rules_plot.apply(
        lambda r: f"{', '.join(sorted(r['antecedents']))[:20]} → {', '.join(sorted(r['consequents']))[:20]}",
        axis=1,
    )

    metrics = ["support", "confidence", "lift"]
    titles = ["Support", "Confidence", "Lift"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        vals = top_rules_plot[metric].values[::-1]
        labels = top_rules_plot["rule_label"].values[::-1]
        colors_bar = plt.cm.plasma(np.linspace(0.2, 0.9, len(vals)))
        axes[idx].barh(range(len(vals)), vals, color=colors_bar)
        axes[idx].set_yticks(range(len(vals)))
        axes[idx].set_yticklabels(labels, fontsize=8)
        axes[idx].set_title(title, fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(title, fontsize=11)
        sns.despine(left=True, ax=axes[idx])

    plt.suptitle("Top-Assoziationsregeln im Vergleich", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_top_regeln_vergleich.png"), dpi=150)
    plt.close()
    print(f"       ✓ 04_top_regeln_vergleich.png")

# 9e. Netzwerk-Graph der Produktassoziationen
if len(rules) > 50:
    # Begrenze auf die 50 stärksten Regeln für bessere Lesbarkeit
    rules_network = rules_sorted.head(50)
else:
    rules_network = rules_sorted

if len(rules_network) > 0:
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sammle alle eindeutigen Produkte
    all_products = set()
    for _, row in rules_network.iterrows():
        all_products.update(row["antecedents"])
        all_products.update(row["consequents"])
    all_products = list(all_products)
    n_products = len(all_products)

    # Zufällige Positionen im 2D-Raum
    np.random.seed(42)
    pos = {p: np.random.randn(2) * 0.8 for p in all_products}

    # Einfache Feder-basierte Positionierung (manuell simuliert)
    for _ in range(50):
        for p in all_products:
            force = np.zeros(2)
            for _, row in rules_network.iterrows():
                if p in row["antecedents"]:
                    for c in row["consequents"]:
                        diff = pos[c] - pos[p]
                        dist = np.linalg.norm(diff) + 0.01
                        force += diff / dist * row["lift"] * 0.01
                if p in row["consequents"]:
                    for a in row["antecedents"]:
                        diff = pos[a] - pos[p]
                        dist = np.linalg.norm(diff) + 0.01
                        force += diff / dist * row["lift"] * 0.01
            pos[p] = pos[p] + force * 0.1

    # Zeichne Kanten (Assoziationen)
    for _, row in rules_network.iterrows():
        for a in row["antecedents"]:
            for c in row["consequents"]:
                ax.plot(
                    [pos[a][0], pos[c][0]],
                    [pos[a][1], pos[c][1]],
                    alpha=min(row["lift"] / 5, 0.8),
                    linewidth=min(row["lift"] * 1.5, 5),
                    color="#2E86AB",
                )

    # Zeichne Knoten (Produkte)
    for p in all_products:
        # Zähle wie oft ein Produkt in Regeln vorkommt (als Maß für Wichtigkeit)
        count = 0
        for _, row in rules_network.iterrows():
            if p in row["antecedents"] or p in row["consequents"]:
                count += 1
        node_size = max(200, count * 80)
        ax.scatter(pos[p][0], pos[p][1], s=node_size, color="#A23B72", zorder=5, edgecolors="white", linewidth=1)
        ax.annotate(p[:30], pos[p], fontsize=7, ha="center", va="center", color="white", fontweight="bold")

    ax.set_title("Assoziations-Netzwerk (Top-Regeln)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_assoziation_netzwerk.png"), dpi=150)
    plt.close()
    print(f"       ✓ 05_assoziation_netzwerk.png")

# ---------------------------------------------------------------------------
# 10. Zusammenfassung
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("  ZUSAMMENFASSUNG")
print(f"{'=' * 70}")
print(f"  Datensatz:             {DATA_PATH}")
print(f"  Land-Filter:           {COUNTRY_FILTER}")
print(f"  Bereinigte Zeilen:     {df.shape[0]:,}")
print(f"  Transaktionen:         {len(transactions):,}")
print(f"  Einzigartige Produkte: {df['StockCode'].nunique():,}")
print(f"  Frequent Itemsets:     {len(frequent_itemsets)}")
print(f"  Assoziationsregeln:    {len(rules)}")
if len(rules) > 0:
    print(f"  Top-Regel:            {{{', '.join(sorted(rules_sorted.iloc[0]['antecedents']))}}} → "
          f"{{{', '.join(sorted(rules_sorted.iloc[0]['consequents']))}}}")
    print(f"  Bester {RANKING_METRIC}:            {rules_sorted.iloc[0][RANKING_METRIC]:.4f}")
print(f"\n  Parameter:")
print(f"    min_support     = {MIN_SUPPORT}")
print(f"    min_confidence  = {MIN_CONFIDENCE}")
print(f"    min_lift        = {MIN_LIFT}")
print(f"\n  Ausgabe: {OUTPUT_DIR}/")
print(f"{'=' * 70}")