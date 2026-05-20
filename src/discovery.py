import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def main():
    print("Starting Extended Process Discovery (Tasks 3 & 4)...")
    
    # 1. Load data
    df = pd.read_pickle("data/processed/df_raw.pkl")
    
    # Create directory for plots
    save_dir = 'models/discovery'
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort by case and timestamp for sequential analysis
    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp']).copy()
    
  # # --- A) TIME-RELATED PERFORMANCE (Top 5 Bottlenecks) ---
    print("1. Calculating Top 5 Bottlenecks...")
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['next_act'] = df_sorted.groupby('case:concept:name')['concept:name'].shift(-1)
    
    # Calculate transition time in days
    df_sorted['transition_time_days'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / (24 * 3600)
    
    transitions = df_sorted.dropna(subset=['next_act']).copy()
    transitions['transition_name'] = transitions['concept:name'] + ' -> ' + transitions['next_act']
    
    # Basis-Statistiken für alle Übergänge berechnen
    transition_stats = transitions.groupby('transition_name')['transition_time_days'].agg(['mean', 'median', 'count'])
    
    # -------------------------------------------------------------------------
    # 1. DIAGRAMM: Sortiert nach MITTELWERT (Original: bottlenecks.png)
    # -------------------------------------------------------------------------
    top_5_mean = transition_stats.sort_values(by='mean', ascending=False).head(5).copy()
    # N in den Anzeigenamen einbauen
    top_5_mean['display_name'] = top_5_mean.index + " (N=" + top_5_mean['count'].astype(str) + ")"
    top_5_mean = top_5_mean.set_index('display_name')
    
    plt.figure(figsize=(13, 5)) # Breite leicht erhöht für die längere Beschriftung
    top_5_mean['mean'].plot(kind='barh', color='#a02c34', edgecolor='black') # Originales Rot
    plt.title('Top 5 Bottlenecks: Mean Transition Durations (N = Number of Cases)')
    plt.xlabel('Average Duration (in Days)')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks.png", bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. DIAGRAMM: Sortiert nach MEDIAN (Neu: bottlenecks_median.png)
    # -------------------------------------------------------------------------
    top_5_median = transition_stats.sort_values(by='median', ascending=False).head(5).copy()
    # N in den Anzeigenamen einbauen
    top_5_median['display_name'] = top_5_median.index + " (N=" + top_5_median['count'].astype(str) + ")"
    top_5_median = top_5_median.set_index('display_name')
    
    plt.figure(figsize=(13, 5))
    top_5_median['median'].plot(kind='barh', color='#a02c34', edgecolor='black') # Alternatives Blau zur Unterscheidung
    plt.title('Top 5 Bottlenecks: Median Transition Durations (N = Number of Cases)')
    plt.xlabel('Median Duration (in Days)')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks_median.png", bbox_inches='tight')
    plt.close()
    
    print("-> Both bottleneck plots generated successfully with case counts (N).")


    # --- B) TOTAL CASE DURATIONS ---
    print("2. Calculating Total Case Durations...")
    case_durations = df_sorted.groupby('case:concept:name').agg(
        start=('time:timestamp', 'min'),
        end=('time:timestamp', 'max')
    )
    case_durations['total_days'] = (case_durations['end'] - case_durations['start']).dt.total_seconds() / (24 * 3600)
    
    plt.figure(figsize=(12, 5))
    # Filter out extreme outliers (> 2 years) for better visualization
    plt.hist(case_durations['total_days'][case_durations['total_days'] < 730], bins=30, color='#a02c34', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Case Durations (Cases < 2 Years)')
    plt.xlabel('Total Duration (in Days)')
    plt.ylabel('Number of Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/case_durations.png", bbox_inches='tight')
    plt.close()

    # --- C) PROCESS COMPLEXITY (Top 5 Variants - Full Paths) ---
    print("3. Analyzing Top 5 Process Variants...")
    # Concatenate activities per case into a full text chain
    variants = df_sorted.groupby('case:concept:name')['concept:name'].apply(lambda x: " -> ".join(x)).reset_index()
    top_variants = variants['concept:name'].value_counts().head(5)
    
    # We use the full label names now (no truncation) to ensure full paths are recognizable
    full_labels = list(top_variants.index)
    
    # Increased figure width to 16 to accommodate long process paths on the Y-axis
    plt.figure(figsize=(16, 6))
    plt.barh(full_labels, top_variants.values, color="#a02c34", edgecolor='black', alpha=0.8)
    plt.title('Top 5 Process Variants (Frequency Distribution)')
    plt.xlabel('Number of Cases')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # tight_layout and bbox_inches='tight' ensure the long text isn't cut off at the edge of the image
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_variants.png", bbox_inches='tight')
    plt.close()

    # --- D) ORGANIZATIONAL PERSPECTIVE ---
    print("4. Checking Organizational Perspective...")
    if 'org:resource' not in df.columns:
        with open(f"{save_dir}/org_info.txt", "w") as f:
            f.write("Limitation: The 4TU Road Traffic Fines dataset does not contain an 'org:resource' column.")
            
    print("✅ All materials for the preliminary presentation successfully saved in 'models/discovery/'!")

# --- E) DOTTED CHART (BATCHING ANALYSIS) ---
    print("5. Generating Dotted Chart (Batching Analysis)...")
    import pm4py
    
    # Sampling: Um Überladung zu vermeiden
    df_sample_dc = df_sorted.head(10000).copy()
    
    # DataFrame für pm4py formatieren (generiert die fehlende @@case_index Spalte)
    df_sample_dc = pm4py.format_dataframe(
        df_sample_dc, 
        case_id='case:concept:name', 
        activity_key='concept:name', 
        timestamp_key='time:timestamp'
    )
    
    # Nutzung der modernen API-Funktion von pm4py
    pm4py.save_vis_dotted_chart(df_sample_dc, f"{save_dir}/dotted_chart.png")
    print("-> Dotted Chart erfolgreich gespeichert.")
    
    # --- F) PERFORMANCE SPECTRUM (TU EINDHOVEN) ---
    print("6. Generating Performance Spectrum...")

    # Für ein sauberes Spectrum filtern wir auf die häufigsten 5 Aktivitäten
    # und nehmen eine Stichprobe von 150 Fällen, sonst wird es ein "Spaghetti-Graph"
    top_acts = df_sorted['concept:name'].value_counts().head(5).index.tolist()
    sample_cases = df_sorted['case:concept:name'].drop_duplicates().sample(150, random_state=42)
    
    df_spectrum = df_sorted[df_sorted['case:concept:name'].isin(sample_cases)].copy()
    df_spectrum = df_spectrum[df_spectrum['concept:name'].isin(top_acts)]
    
    # Y-Achsen Mapping für die Aktivitäten (Reihenfolge erzwingen)
    act_y = {act: i for i, act in enumerate(top_acts)}
    
    plt.figure(figsize=(14, 7))
    
    # Zeichne die Verbindungslinien pro Fall
    for case, group in df_spectrum.groupby('case:concept:name'):
        if len(group) > 1:
            group = group.sort_values('time:timestamp')
            x = group['time:timestamp']
            y = group['concept:name'].map(act_y)
            # Flache Linien = schnell, Steile Linien = lange Wartezeit
            plt.plot(x, y, marker='o', markersize=4, alpha=0.5, linewidth=1.5)

    plt.yticks(range(len(top_acts)), top_acts)
    plt.title('Performance Spectrum (Sample of 150 Cases over Time)')
    plt.xlabel('Zeitachse (2000 - 2013)')
    plt.ylabel('Prozessschritt')
    
    # Formatierung der X-Achse für bessere Lesbarkeit der Jahre
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_spectrum.png", bbox_inches='tight')
    plt.close()
    print("-> Performance Spectrum erfolgreich gespeichert.")

if __name__ == "__main__":
    main()