import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pm4py
from sklearn.model_selection import train_test_split
import os

def main():
    print("--- STARTE EVALUIERUNG ---\n")
    
    # 1. SHAP EVALUATION FÜR RANDOM FOREST
    print("1. Generiere SHAP Plot für Random Forest...")
    X_rf = pd.read_pickle("data/processed/X_rf.pkl")
    y_rf = pd.read_pickle("data/processed/y_rf.pkl")
    rf_model = joblib.load('models/rf_model.pkl')
    
    # Gleicher Split wie beim Training
    _, X_test, _, _ = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)
    
    X_test_sample = X_test.sample(1000, random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure()
    plt.title("SHAP Summary Plot")
    
    # Check für die SHAP-Werte Struktur (kann je nach Version variieren)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test_sample, plot_type="dot", show=False)
    else:
        shap.summary_plot(shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values, X_test_sample, plot_type="dot", show=False)
        
    plt.savefig("models/shap_summary.png", bbox_inches='tight')
    print("✅ SHAP Plot als 'models/shap_summary.png' gespeichert.")

    # 2. CONFORMANCE CHECKING
    print("\n2. Starte Conformance Checking...")
    df = pd.read_pickle("data/processed/df_raw.pkl")
    
    # Top 10 Varianten filtern für den "Happy Path"
    happy_path_log = pm4py.filter_variants_top_k(df, 10)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(happy_path_log)
    
    sample_cases = df['case:concept:name'].drop_duplicates().sample(2000, random_state=42)
    real_world_sample = df[df['case:concept:name'].isin(sample_cases)]
    
    fitness = pm4py.fitness_token_based_replay(real_world_sample, net, initial_marking, final_marking)
    print(f"✅ Prozent der Fälle mit perfektem Durchlauf: {fitness['perc_fit_traces']:.2f} %")

if __name__ == "__main__":
    main()