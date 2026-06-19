---
name: explain-code
description: Explains code step-by-step using analogies, ASCII diagrams, and identifying common pitfalls. Use when the user asks "explain this code", "what does this script do", "how does this work", "walk me through this", or any request to understand a code file. Always activate when a user pastes code or references a script file and asks for an explanation.
---

# Explain Code

When asked to explain a piece of code, follow this structure **in order**:

## 1. Analogie aus dem Alltag

Finde eine passende Analogie, die den Code mit etwas aus dem täglichen Leben vergleicht. Die Analogie soll dem Leser helfen, den **Zweck** des Codes auf einer konzeptionellen Ebene zu verstehen, bevor er ins Detail geht.

- Wähle eine Analogie, die zur **Komplexität** des Codes passt (Kochen, Poststelle, Fabrikfließband, Bibliothek, Verkehr, etc.)
- Erkläre kurz den Zusammenhang zwischen den Analogie-Elementen und den Code-Komponenten

## 2. ASCII-Diagramm

Zeichne den Ablauf, die Struktur oder die Zusammenhänge als ASCII-Art-Diagramm. Das Diagramm soll die **Architektur oder den Datenfluss** visualisieren.

- Verwende `+--`, `|`, `v`, `^` für Verbindungen
- Zeige Funktionen, Datenflüsse und Entscheidungspunkte
- Halte es sauber und lesbar (~10-25 Zeilen)

## 3. Schritt-für-Schritt-Erklärung

Gehe den Code in logischen Blöcken durch. Erkläre **jede signifikante Zeile** oder zumindest jeden Codeblock.

- Orientiere dich an der Reihenfolge der Ausführung (nicht an der Datei-Reihenfolge)
- Erkläre **was** jeder Teil tut und **warum** er es tut
- Hebe ungewöhnliche oder idiomatische Konstrukte hervor
- Nenne Datei und Zeilennummern aus der zu erklärenden Datei

## 4. Stolperstein

Identifiziere **einen** spezifischen Stolperstein – etwas, das häufig falsch verstanden wird oder zu Bugs führt.

- Nenne den konkreten Fehler oder das Missverständnis
- Zeige, warum er gerade in diesem Code besonders relevant ist
- Gib ggf. einen Tipp, wie man ihn vermeidet