import os
import pm4py
import pandas as pd

def main():
    print("Starte Data Preparation...")
    
    # 1. Pfad zur XES Datei (Passe den Dateinamen an, falls er bei dir anders heißt!)
    xes_path = "data/raw/Road_Traffic_Fine_Management_Process.xes.gz" 
    
    if not os.path.exists(xes_path):
        # Falls es nicht gezippt ist, probiere die normale .xes
        xes_path = "data/raw/Road_Traffic_Fine_Management_Process.xes"
        if not os.path.exists(xes_path):
            raise FileNotFoundError(f"Die Datei {xes_path} wurde nicht gefunden!")

    # 2. Daten laden
    print(f"Lese Event Log ein von: {xes_path}")
    event_log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(event_log)
    
    # 3. Fall-Ausgang bestimmen (Labeling)
    print("Labeling der Fälle (Payment = 0, Inkasso = 1)...")
    cases = df.groupby('case:concept:name')['concept:name'].apply(list).reset_index()

    def determine_outcome(activity_list):
        if 'Payment' in activity_list: return 0
        elif 'Send for Credit Collection' in activity_list: return 1
        else: return -1

    cases['label'] = cases['concept:name'].apply(determine_outcome)
    completed_cases = cases[cases['label'] != -1].copy()

    # 4. Daten speichern (als .pkl für schnelles Laden in den nächsten Schritten)
    os.makedirs('data/processed', exist_ok=True)
    df.to_pickle("data/processed/df_raw.pkl")
    completed_cases.to_pickle("data/processed/completed_cases.pkl")
    
    print(f"Fertig! {len(completed_cases)} abgeschlossene Fälle gespeichert in data/processed/")

if __name__ == "__main__":
    main()