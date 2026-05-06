# 🚦 Predictive Process Analytics: Road Traffic Fines

Dieses Projekt prognostiziert für laufende Bußgeldverfahren (*Road Traffic Fine Management Process*), ob diese regulär bezahlt werden oder im Inkasso landen.

## 1. Setup & Installation
Lade das Projekt herunter und installiere die benötigten Pakete in deiner virtuellen Umgebung:

```bash
git clone [https://github.com/DEIN_NAME/traffic-fine-prediction.git](https://github.com/DEIN_NAME/traffic-fine-prediction.git)
cd traffic-fine-prediction
pip install -r requirements.txt

2. Daten-Pipeline starten
Führt die gesamte Datenbereinigung, das Feature Engineering, das Modell-Training (Random Forest & LSTM) und die Evaluierung automatisch aus:

Bash
python run_pipeline.py
(Hinweis: Die originale Datensatz-Datei muss dafür vorab im Ordner data/raw/ platziert werden.)

3. Dashboard starten
Startet die Streamlit Web-App zur interaktiven Analyse einzelner Fälle:
-------
Bash
streamlit run app/app.py
Das Dashboard öffnet sich anschließend automatisch im Browser unter http://localhost:8501.
-------

👥 Team
Lennard Ruf
Markus Schneele
Ali Hawash
Krzysztof Olesiak

