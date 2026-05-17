import pandas as pd
import matplotlib.pyplot as plt
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
    
    # --- A) TIME-RELATED PERFORMANCE (Top 5 Bottlenecks) ---
    print("1. Calculating Top 5 Bottlenecks...")
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['next_act'] = df_sorted.groupby('case:concept:name')['concept:name'].shift(-1)
    
    # Calculate transition time in days
    df_sorted['transition_time_days'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / (24 * 3600)
    
    transitions = df_sorted.dropna(subset=['next_act']).copy()
    transitions['transition_name'] = transitions['concept:name'] + ' -> ' + transitions['next_act']
    
    # Aggregate and filter for top 5 bottlenecks
    avg_transitions = transitions.groupby('transition_name')['transition_time_days'].mean().sort_values(ascending=False).head(5)
    
    plt.figure(figsize=(12, 5))
    avg_transitions.plot(kind='barh', color='#a02c34', edgecolor='black')
    plt.title('Top 5 Bottlenecks: Transition Durations between Process Steps')
    plt.xlabel('Average Duration (in Days)')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks.png", bbox_inches='tight')
    plt.close()

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

if __name__ == "__main__":
    main()