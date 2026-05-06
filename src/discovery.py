import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("Starte Process Discovery (Task 4)...")
    
    # 1. Daten laden
    df = pd.read_pickle("data/processed/df_raw.pkl")
    
    # Ordner für die Diagramme erstellen
    save_dir = 'models/discovery'
    os.makedirs(save_dir, exist_ok=True)
    
    # --- A) TIME-RELATED PERFORMANCE (Bottleneck Analyse) ---
    print("Berechne Bottlenecks (Übergangszeiten)...")
    
    # Wir sortieren nach Fall und Zeit
    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp']).copy()
    
    # Wir holen uns die jeweils nächste Aktivität und deren Zeitstempel
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['next_act'] = df_sorted.groupby('case:concept:name')['concept:name'].shift(-1)
    
    # Dauer in Tagen berechnen
    df_sorted['transition_time_days'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / (24 * 3600)
    
    # Übergänge definieren (z.B. "Create Fine -> Send Fine")
    transitions = df_sorted.dropna(subset=['next_act']).copy()
    transitions['transition_name'] = transitions['concept:name'] + ' ➡️ ' + transitions['next_act']
    
    # Durchschnittliche Dauer pro Übergang berechnen
    avg_transitions = transitions.groupby('transition_name')['transition_time_days'].mean().sort_values(ascending=False).head(10)
    
    # Plot: Top 10 Bottlenecks
    plt.figure(figsize=(10, 6))
    avg_transitions.plot(kind='barh', color='#d62728', edgecolor='black')
    plt.title('Top 10 Bottlenecks: Dauer zwischen Prozessschritten')
    plt.xlabel('Durchschnittliche Dauer (in Tagen)')
    plt.ylabel('')
    plt.gca().invert_yaxis() # Längster Balken nach oben
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks.png")
    plt.close()
    
    # --- B) ORGANIZATIONAL PERSPECTIVE ---
    print("Prüfe Organisations-Perspektive...")
    if 'org:resource' in df.columns:
        print("Ressourcen gefunden. (Könnte hier geplottet werden)")
    else:
        # Datensatz enthält keine Ressourcen, wir speichern eine Info ab
        with open(f"{save_dir}/org_info.txt", "w") as f:
            f.write("Limitation: Der 4TU Road Traffic Fines Datensatz enthält keine Spalte für 'org:resource' (Mitarbeiter/Systeme). Eine organisatorische Mining-Analyse ist daher mit diesen Rohdaten nicht durchführbar.")
            
    print("✅ Process Discovery Diagramme gespeichert!")

if __name__ == "__main__":
    main()