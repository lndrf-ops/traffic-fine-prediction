"""Task 4: Process Discovery
- Workflow perspective: Process Complexity, Anomalies, Variant Analysis
- Time/Performance perspective: Bottleneck Analysis
- Performance Spectrum (TU Eindhoven)
- Organizational perspective
- Saves: outputs/plots/bottlenecks*.png, top_variants.png, performance_spectrum.png, petri_net.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def main():
    print("=" * 60)
    print("TASK 4: Process Discovery")
    print("=" * 60)

    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    save_dir = 'outputs/plots'
    os.makedirs(save_dir, exist_ok=True)

    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp']).copy()

    # --- A) BOTTLENECK ANALYSIS ---
    print("  1. Calculating Bottlenecks...")
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['next_act'] = df_sorted.groupby('case:concept:name')['concept:name'].shift(-1)
    df_sorted['transition_time_days'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / (24 * 3600)

    transitions = df_sorted.dropna(subset=['next_act']).copy()
    transitions['transition_name'] = transitions['concept:name'] + ' -> ' + transitions['next_act']

    transition_stats = transitions.groupby('transition_name')['transition_time_days'].agg(['mean', 'median', 'count'])

    # Mean Bottlenecks
    top_5_mean = transition_stats.sort_values(by='mean', ascending=False).head(5).copy()
    top_5_mean['display_name'] = top_5_mean.index + " (N=" + top_5_mean['count'].astype(str) + ")"
    top_5_mean = top_5_mean.set_index('display_name')

    plt.figure(figsize=(13, 5))
    top_5_mean['mean'].plot(kind='barh', color='#a02c34', edgecolor='black')
    plt.title('Top 5 Bottlenecks: Mean Transition Durations')
    plt.xlabel('Average Duration (Days)')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks_mean.png", bbox_inches='tight')
    plt.close()

    # Median Bottlenecks
    top_5_median = transition_stats.sort_values(by='median', ascending=False).head(5).copy()
    top_5_median['display_name'] = top_5_median.index + " (N=" + top_5_median['count'].astype(str) + ")"
    top_5_median = top_5_median.set_index('display_name')

    plt.figure(figsize=(13, 5))
    top_5_median['median'].plot(kind='barh', color='#a02c34', edgecolor='black')
    plt.title('Top 5 Bottlenecks: Median Transition Durations')
    plt.xlabel('Median Duration (Days)')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bottlenecks_median.png", bbox_inches='tight')
    plt.close()
    print(f"     ✅ Bottleneck plots saved")

    # --- B) CASE DURATIONS ---
    print("  2. Calculating Case Durations...")
    case_durations = df_sorted.groupby('case:concept:name').agg(
        start=('time:timestamp', 'min'),
        end=('time:timestamp', 'max')
    )
    case_durations['total_days'] = (case_durations['end'] - case_durations['start']).dt.total_seconds() / (24 * 3600)

    plt.figure(figsize=(12, 5))
    # Filter to < 730 days (2 years) to exclude extreme outliers and focus on the main distribution
    plt.hist(case_durations['total_days'][case_durations['total_days'] < 730], bins=30, color='#a02c34', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Case Durations (< 2 Years)')
    plt.xlabel('Total Duration (Days)')
    plt.ylabel('Number of Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/case_durations.png", bbox_inches='tight')
    plt.close()
    print(f"     ✅ Case duration plot saved")

    # --- C) VARIANT ANALYSIS ---
    print("  3. Analyzing Top 5 Process Variants...")
    variants = df_sorted.groupby('case:concept:name')['concept:name'].apply(lambda x: " -> ".join(x)).reset_index()
    top_variants = variants['concept:name'].value_counts().head(5)

    plt.figure(figsize=(16, 6))
    plt.barh(list(top_variants.index), top_variants.values, color="#a02c34", edgecolor='black', alpha=0.8)
    plt.title('Top 5 Process Variants (Frequency)')
    plt.xlabel('Number of Cases')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_variants.png", bbox_inches='tight')
    plt.close()
    print(f"     ✅ Variant analysis plot saved")

    # --- D) PERFORMANCE SPECTRUM ---
    print("  4. Generating Performance Spectrum...")
    top_acts = df_sorted['concept:name'].value_counts().head(5).index.tolist()
    # 150 cases: enough to see patterns without overplotting the spectrum
    sample_cases = df_sorted['case:concept:name'].drop_duplicates().sample(150, random_state=42)
    df_spectrum = df_sorted[df_sorted['case:concept:name'].isin(sample_cases)].copy()
    df_spectrum = df_spectrum[df_spectrum['concept:name'].isin(top_acts)]

    act_y = {act: i for i, act in enumerate(top_acts)}

    plt.figure(figsize=(14, 7))
    for case, group in df_spectrum.groupby('case:concept:name'):
        if len(group) > 1:
            group = group.sort_values('time:timestamp')
            x = group['time:timestamp']
            y = group['concept:name'].map(act_y)
            plt.plot(x, y, marker='o', markersize=4, alpha=0.5, linewidth=1.5)

    plt.yticks(range(len(top_acts)), top_acts)
    plt.title('Performance Spectrum (150 Cases)')
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_spectrum.png", bbox_inches='tight')
    plt.close()
    print(f"     ✅ Performance spectrum saved")

    # --- E) PETRI NET (Process Model) ---
    print("  5. Discovering Petri Net...")
    happy_path_log = pm4py.filter_variants_top_k(df, 10)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(happy_path_log)

    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    gviz.graph_attr['rankdir'] = 'LR'
    gviz.graph_attr['bgcolor'] = 'transparent'
    gviz.node_attr['color'] = '#a02c34'
    gviz.node_attr['fontcolor'] = '#a02c34'
    gviz.node_attr['fontname'] = 'Arial'
    gviz.edge_attr['color'] = '#a02c34'
    gviz.edge_attr['fontname'] = 'Arial'

    pn_visualizer.save(gviz, f"{save_dir}/petri_net.png")
    print(f"     ✅ Petri Net saved")

    # Save DOT source
    dot_string = gviz.source if hasattr(gviz, 'source') else str(gviz)
    with open(f"{save_dir}/petri_net.dot", "w", encoding="utf-8") as f:
        f.write(dot_string)

    # Note: Organizational perspective (resource analysis) is covered in t3b_batching_analysis.

    print("  ✅ Process Discovery complete")


if __name__ == "__main__":
    main()
