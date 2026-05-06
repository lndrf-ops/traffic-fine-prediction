import time
from src import data_prep, features, train, evaluate, discovery

def run_all():
    start_time = time.time()
    print("🚀 STARTE DIE GESAMTE DATA SCIENCE PIPELINE...\n" + "="*50)

    try:
        print("\n▶️ SCHRITT 1: Data Preparation")
        data_prep.main()

        print("\n▶️ SCHRITT 2: Feature Engineering")
        features.main()

        print("\n▶️ SCHRITT 3: Modell-Training")
        train.main()

        print("\n▶️ SCHRITT 4: Evaluierung & SHAP")
        evaluate.main()
        
        print("\n▶️ SCHRITT 5: Process Discovery")
        discovery.main()

        end_time = time.time()
        duration = (end_time - start_time) / 60
        print("\n" + "="*50)
        print(f"✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN in {duration:.2f} Minuten!")

    except Exception as e:
        print("\n❌ FEHLER IN DER PIPELINE!")
        print(f"Der Prozess wurde abgebrochen wegen: {e}")

if __name__ == "__main__":
    run_all()