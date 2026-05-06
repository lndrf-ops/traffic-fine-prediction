# 🚦 Predictive Process Analytics: Road Traffic Fines

Dieses Projekt bietet eine End-to-End-Lösung zur Analyse und Vorhersage des *Road Traffic Fine Management Process*. Es kombiniert klassisches Process Mining mit modernem Deep Learning und Generative AI, um Zahlungsausfälle frühzeitig zu erkennen und proaktiv zu verhindern.

## 🌟 Key Features & Bonus Tasks
* **Predictive Analytics (Task 5):** Vergleich von Random Forest (Baseline) und LSTM (Deep Learning) Modellen.
* **Prescriptive Modeling (Bonus 3):** Dynamische Handlungsempfehlungen für Sachbearbeiter basierend auf dem Vorhersagerisiko.
* **Process Discovery & Bottlenecks (Task 4):** Interaktive Analyse der Zeitfresser und Workflow-Modellierung (Petri-Netze).
* **Explainable AI (Bonus 2):** Post-hoc Erklärungen der Modellentscheidungen mittels SHAP-Werten.
* **Generative AI (Bonus 5):** Erzeugung synthetischer Event-Logs mittels stochastischer Markov-Ketten.
* **Advanced Web-App (Bonus 7):** Deployment als interaktives Streamlit-Dashboard mit Plotly-Visualisierungen.

## 1. Setup & Installation
Lade das Projekt herunter und installiere die benötigten Pakete in deiner virtuellen Umgebung:

git clone [https://github.com/lndrf-ops/traffic-fine-prediction](https://github.com/lndrf-ops/traffic-fine-prediction)
cd traffic-fine-prediction
pip install -r requirements.txt

## 2. Daten-Pipeline starten
Führt die gesamte Datenbereinigung, das Feature Engineering, das Modell-Training (Random Forest & LSTM) und die Evaluierung automatisch aus:

Bash
python run_pipeline.py
(Hinweis: Die originale Datensatz-Datei muss dafür vorab im Ordner data/raw/ platziert werden.)

## 3. Dashboard starten
Startet die Streamlit Web-App zur interaktiven Analyse einzelner Fälle:
-------
Bash
streamlit run app/app.py
Das Dashboard öffnet sich anschließend automatisch im Browser unter http://localhost:8501.
-------

## Team
Lennard Ruf
Markus Schneele
Ali Hawash
Krzysztof Olesiak

