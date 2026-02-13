# Genetic Algorithm - Schnellstart Anleitung (Deutsch)

## Überblick

Der `run_ga.py` Skript ist dein vorkonfigurierter "Run-Button" zum Starten des Genetischen Algorithmus für die Entwicklung von Handelsstrategien. Das Skript automatisiert den gesamten Evolutionsprozess und gibt am Ende die Top 5 erfolgreichsten Strategien aus.

## Schnellstart

### Einfacher Start (empfohlen)

```bash
python genetic_algorithm/run_ga.py
```

Dies wird:
- 50 Strategien über 20 Generationen entwickeln
- Die Top 5 besten Strategien anzeigen
- Strategien nach `genetic_algorithm/output/` speichern
- Einen Zusammenfassungsbericht erstellen

### Schnelle Demo (5 Minuten)

```bash
python genetic_algorithm/demo_ga_runner.py
```

Dies führt eine Minimal-Version mit nur 5 Strategien und 2 Generationen aus.

## Konfiguration

Am Anfang von `run_ga.py` findest du die **USER CONFIGURATION** Sektion:

```python
# Basis GA Parameter
POPULATION_SIZE = 50          # Anzahl Strategien pro Generation
GENERATIONS = 20              # Anzahl der Generationen
MUTATION_RATE = 0.15          # Mutationswahrscheinlichkeit (0.0-1.0)
CROSSOVER_RATE = 0.7          # Crossover-Wahrscheinlichkeit (0.0-1.0)
ELITE_SIZE = 5                # Anzahl der besten Strategien zum Bewahren

# Anzahl der Top-Strategien zum Anzeigen und Speichern
TOP_STRATEGIES_COUNT = 5

# Output Konfiguration
SAVE_STRATEGIES = True        # Top-Strategien in Dateien speichern
OUTPUT_DIR = Path("genetic_algorithm/output")
```

### Empfohlene Einstellungen

**Für schnelle Tests (5-10 Minuten):**
```python
POPULATION_SIZE = 20
GENERATIONS = 10
```

**Für normale Läufe (30-60 Minuten):**
```python
POPULATION_SIZE = 50
GENERATIONS = 20
```

**Für intensive Suche (mehrere Stunden):**
```python
POPULATION_SIZE = 100
GENERATIONS = 50
```

## Was passiert während eines Laufs?

1. **Initialisierung**
   - Lädt Konfiguration
   - Zeigt aktuelle Einstellungen
   - Wartet auf Bestätigung (Enter drücken)

2. **Evolutions-Loop** (für jede Generation)
   - Bewertet alle Strategien via Backtesting
   - Wählt beste Performer aus
   - Erstellt Nachkommen via Crossover und Mutation
   - Bewahrt Elite-Strategien

3. **Ergebnisse**
   - Zeigt Top 5 Strategien mit Metriken an
   - Speichert Strategie Python-Dateien
   - Erstellt Zusammenfassungsbericht

## Output-Dateien

Nach einem erfolgreichen Lauf findest du:

### Strategie-Dateien
```
genetic_algorithm/output/strategy_rank1_genX_indY_TIMESTAMP.py
```
- Fertig zum Verwenden mit FreqTrade
- Können nach `user_data/strategies/` kopiert werden

### Zusammenfassungsbericht
```
genetic_algorithm/output/ga_summary_TIMESTAMP.txt
```
- Übersicht über den Lauf
- Top-Strategien mit Metriken

### Log-Datei
```
genetic_algorithm/logs/ga_run_TIMESTAMP.log
```
- Detailliertes Ausführungsprotokoll
- Nützlich für Debugging

## Strategie-Metriken verstehen

Jede Strategie wird anhand mehrerer Metriken bewertet:

- **Fitness Score**: Gesamtqualität (0-1, höher ist besser)
- **Profit**: Gesamtrendite in Prozent (Ziel: 10-50%+)
- **Sharpe Ratio**: Risikobereinigter Ertrag (Ziel: 1.0+, exzellent: 2.0+)
- **Max Drawdown**: Größter Peak-to-Trough Rückgang (Ziel: < 20%)
- **Win Rate**: Prozentsatz profitabler Trades (Ziel: > 50%)
- **Total Trades**: Anzahl ausgeführter Trades (Ziel: 20-50)
- **Profit Factor**: Bruttogewinn / Bruttoverlust (Ziel: > 1.5)

## Nächste Schritte nach GA-Lauf

1. **Ergebnisse überprüfen**
   - Überprüfe die angezeigten Top-Strategien
   - Überprüfe Metriken und Parameter

2. **Backtest mit mehr Daten**
   ```bash
   # Kopiere Strategie nach user_data/strategies/
   cp genetic_algorithm/output/strategy_rank1_*.py user_data/strategies/
   
   # Führe FreqTrade Backtest aus
   freqtrade backtesting --strategy <StrategyName>
   ```

3. **Test im Dry-Run**
   ```bash
   freqtrade trade --dry-run --strategy <StrategyName>
   ```

4. **Performance validieren**
   - Führe mehrere Tage im Dry-Run aus
   - Überprüfe ob Performance mit Backtests übereinstimmt

5. **Für Live-Trading bereitstellen** (nur wenn überzeugt)
   ```bash
   freqtrade trade --strategy <StrategyName>
   ```

## Tipps für beste Ergebnisse

1. **Klein anfangen**: Verwende kleinere Population und weniger Generationen für erste Tests
2. **Iterieren**: Führe mehrere Male mit verschiedenen Konfigurationen aus
3. **Diversifizieren**: Probiere verschiedene Fitness-Weight-Kombinationen
4. **Validieren**: Teste Top-Strategien immer mit mehr Daten
5. **Geduldig sein**: Gute Strategien brauchen Zeit zum Entwickeln

## Beispiel-Workflow

```bash
# Schneller Test (5-10 Minuten)
# Bearbeite run_ga.py: POPULATION_SIZE=20, GENERATIONS=10
python genetic_algorithm/run_ga.py

# Voller Lauf (30-60 Minuten)
# Bearbeite run_ga.py: POPULATION_SIZE=50, GENERATIONS=20
python genetic_algorithm/run_ga.py

# Intensive Suche (mehrere Stunden)
# Bearbeite run_ga.py: POPULATION_SIZE=100, GENERATIONS=50
python genetic_algorithm/run_ga.py
```

## Problembehebung

### "Configuration file not found"
- Stelle sicher, dass du vom Repository-Root ausführst
- Überprüfe dass `genetic_algorithm/config/ga_config.yaml` existiert

### Evolution ist sehr langsam
- Reduziere POPULATION_SIZE (versuche 20-30)
- Reduziere GENERATIONS (versuche 10-15)

### Alle Strategien haben niedrige Fitness
- Überprüfe Backtesting-Konfiguration
- Verifiziere dass Datendateien in tests/testdata existieren

## Zusätzliche Dokumentation

Für mehr Details siehe:
- **RUN_GA_GUIDE.md** - Vollständige englische Anleitung
- **README.md** - Hauptdokumentation
- **TUTORIAL.md** - Vollständige Nutzungsanleitung

## Zusammenfassung

Mit `run_ga.py` hast du einen einfachen "Run-Button" um den GA zu starten:

1. Bearbeite die Konfiguration am Anfang der Datei (optional)
2. Führe `python genetic_algorithm/run_ga.py` aus
3. Drücke Enter zum Starten
4. Warte bis Evolution abgeschlossen ist
5. Überprüfe die Top 5 Strategien in der Ausgabe
6. Finde gespeicherte Strategien in `genetic_algorithm/output/`

Viel Erfolg! 🚀
