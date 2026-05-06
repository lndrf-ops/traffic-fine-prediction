# src/generative_ai.py
import pandas as pd
import random
import os

def build_transition_matrix(sequences):
    """Baut ein Wahrscheinlichkeits-Wörterbuch (Markov Modell) aus echten Daten."""
    transitions = {}
    for seq in sequences:
        for i in range(len(seq)):
            current_state = seq[i]
            # Wenn es das letzte Event ist, markieren wir das Ende
            next_state = seq[i+1] if i + 1 < len(seq) else 'END'
            
            if current_state not in transitions:
                transitions[current_state] = []
            transitions[current_state].append(next_state)
    return transitions

def generate_trace(transitions, start_state='Create Fine', max_length=20):
    """Generiert einen künstlichen Trace basierend auf den gelernten Wahrscheinlichkeiten."""
    current_state = start_state
    trace = [current_state]
    
    for _ in range(max_length - 1):
        if current_state not in transitions:
            break
        
        # Wähle den nächsten Schritt zufällig basierend auf der echten Häufigkeit
        next_state = random.choice(transitions[current_state])
        
        if next_state == 'END':
            break
            
        trace.append(next_state)
        current_state = next_state
        
    return trace

def main():
    print("Starte Generative AI (Bonus 5)...")
    
    # 1. Lade echte Sequenzen
    completed_cases = pd.read_pickle("data/processed/completed_cases.pkl")
    real_sequences = completed_cases['concept:name'].tolist()
    
    # 2. Trainiere das Generative Markov-Modell
    print("Trainiere stochastisches Modell (Markov Chain) auf echten Daten...")
    transition_matrix = build_transition_matrix(real_sequences)
    
    # 3. Generiere synthetischen Event Log
    num_synthetic_cases = 1000
    print(f"Generiere {num_synthetic_cases} synthetische Fälle...")
    
    synthetic_log = []
    for i in range(num_synthetic_cases):
        syn_trace = generate_trace(transition_matrix, start_state='Create Fine')
        
        # In Format für Event Log bringen
        for step, activity in enumerate(syn_trace):
            synthetic_log.append({
                'case:concept:name': f"SYNTH_{i+1}",
                'concept:name': activity,
                'event_position': step + 1
            })
            
    df_synthetic = pd.DataFrame(synthetic_log)
    
    # 4. Speichern
    save_path = "data/processed/synthetic_event_log.csv"
    df_synthetic.to_csv(save_path, index=False)
    print(f"✅ Synthetischer Event Log mit {len(df_synthetic)} Events erfolgreich generiert!")
    print(f"   Gespeichert unter: {save_path}")

if __name__ == "__main__":
    main()