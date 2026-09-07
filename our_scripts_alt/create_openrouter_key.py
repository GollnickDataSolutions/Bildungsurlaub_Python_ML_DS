"""
Erstelle neue API-Keys bei OpenRouter via Python.
Voraussetzung: Du brauchst einen bestehenden API-Key mit Admin-Rechten.

Dokumentation: https://openrouter.ai/docs/api-keys
"""
#%%
import requests
import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()  # Lädt Umgebungsvariablen aus .env-Datei
#%%
# -- Konfiguration --
EXISTING_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not EXISTING_API_KEY:
    print("❌ Kein API-Key gefunden.")
    print("   Setze die Umgebungsvariable OPENROUTER_API_KEY oder trage den Key direkt unten ein.")
    sys.exit(1)
#%%
# OpenRouter API Endpunkt
url = "https://openrouter.ai/api/v1/keys"

# Nur DeepSeek-Modelle sind für diesen Key erlaubt
payload = {
    "name": "Mein neuer Python-Key2",          # Name/Label
    "limit": 5.00,                             # Optional: Kostenlimit in USD
    "models": ["deepseek/"],                   # Erlaubt nur Modelle von DeepSeek
    "disabled": False                          # Optional: Key deaktivieren
}

headers = {
    "Authorization": f"Bearer {EXISTING_API_KEY}",
    "Content-Type": "application/json"
}

print("🚀 Erstelle neuen API-Key bei OpenRouter...")
response = requests.post(url, json=payload, headers=headers)

if response.status_code == 201:
    new_key = response.json()
    print("\n✅ Neuer API-Key erstellt:")
    print(f"   Name: {new_key.get('name')}")
    print(f"   Key:  {new_key.get('key')}")
    print(f"   Hash: {new_key.get('hash')}")
    print(f"   Erstellt: {new_key.get('created_at')}")
    print("\n⚠️  Speichere den Key sicher ab — er wird nur einmal angezeigt!")
else:
    print(f"\n❌ Fehler {response.status_code}: {response.text}")
    print("\nHinweise:")
    print("  - Dein bestehender Key muss Admin-Rechte haben")
    print("  - Oder erstelle Keys manuell unter https://openrouter.ai/keys")
# %%
