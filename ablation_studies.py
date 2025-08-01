"""
Ablation Study: Adaptive vs. Non-Adaptive Analysis Approach Performance (Step 3)

Description:
    This script conducts an ablation study to compare the performance of the
    multi-adapter pedestrian analysis approach under two different operational modes.
    It uses the raw data generated by the inference scripts (Step 2) to simulate
    the computational cost (GFLOPs) and latency (inference time) of the approach.

Operational Modes:
    -   `ALL` Mode (Baseline):
        This mode simulates a non-adaptive, brute-force approach. It assumes all
        behavioral adapters are fired for every pedestrian in every frame, and all
        scene-level adapters are fired periodically (e.g., every 30 frames).
        This represents the maximum computational cost and information gain.

    -   `ADAPTIVE` Mode (Optimized):
        This mode simulates an intelligent, rule-based system that selectively
        fires adapters based on the current scene context. The rules are designed
        to minimize redundant computations by only activating adapters when the
        environment (e.g., weather, time of day, pedestrian density) justifies
        their use.

The pipeline operates as follows:
    1.  Loads the raw scene context and pedestrian behavior data from the CSV
        files generated in Step 2.
    2.  Calculates data-driven default values for adapter GFLOPs and inference
        times to handle any missing data points.
    3.  Simulates the `ALL` mode to establish a performance baseline.
    4.  Simulates the `ADAPTIVE` mode using the predefined decision rules.
    5.  Generates a comparative bar plot (`adaptive_vs_all_ablation.png`) to
        visualize the performance differences.
    6.  Generates a detailed markdown report (`ablation_report.md`) with a
        statistical breakdown of the results, including percentage changes.

Inputs:
    -   Scene Context Data: Path specified by `--scene_csv`.
        (Default: `raw_data_scene_contextual_analysis_step2.csv`)
    -   Pedestrian Behavior Data: Path specified by `--behavior_csv`.
        (Default: `raw_data_pedestrian_behavior_analysis_step2.csv`)

Outputs:
    -   Ablation Study Plot: `Ablation_Studies_Report/adaptive_vs_all_ablation.png`
    -   Ablation Study Report: `Ablation_Studies_Report/ablation_report.md`

Command-Line Arguments:
    -   `--scene_csv` (str, optional):
        Path to the CSV file containing the scene contextual analysis results.
    -   `--behavior_csv` (str, optional):
        Path to the CSV file containing the pedestrian behavior analysis results.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# =============================================================================
# A. GLOBAL CONFIGURATION 
# =============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
COLORS = ["#004488", "#DDAA33", "#BB5566", "#FF5733"]
ABLATION_DIR = Path("Ablation_Studies_Report")

# =============================================================================
# B: DATA LOADING & DEFAULTS CALCULATION
# =============================================================================

def load_ablation_data(scene_csv: str, behavior_csv: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Loads and prepares the scene and behavior data for the ablation study."""

    print("Loading data for ablation studies...")
    if not os.path.exists(scene_csv):
        raise FileNotFoundError(f"Scene data CSV not found at: {scene_csv}")
    
    scene_df = pd.read_csv(scene_csv)
    print(f"   - Scene data loaded: {len(scene_df):,} rows")

    behavior_df = None
    if behavior_csv and os.path.exists(behavior_csv):
        behavior_df = pd.read_csv(behavior_csv)
        print(f"   - Behavior data loaded: {len(behavior_df):,} rows")
    else:
        print("   - Warning: Behavior data CSV not found or not provided. Some metrics may be unavailable.")

    return scene_df, behavior_df

def calculate_default_metrics(scene_df: pd.DataFrame, behavior_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculates average inference time and GFLOPs from the raw data. These values
    are used as fallbacks for any adapters where performance metrics are missing.
    """

    print("\nCalculating data-driven default values for performance metrics...")

    BEHAVIOR_ADAPTERS = ["ACTION", "LOOK", "CROSS", "OCCLUSION"]
    SCENE_ADAPTERS = ["WEATHER", "TIME_OF_DAY", "PED_DENSITY", "ROAD_PRESENCE"]

    all_gflops_series = []
    if behavior_df is not None:
        for adapter in BEHAVIOR_ADAPTERS:
            gflops_col = f"gflops_{adapter.lower()}"
            if gflops_col in behavior_df.columns:
                all_gflops_series.append(behavior_df[gflops_col].dropna())

    for adapter in SCENE_ADAPTERS:
        gflops_col = f"gflops_{adapter.lower()}"
        if gflops_col in scene_df.columns:
            all_gflops_series.append(scene_df[gflops_col].dropna())

    mean_gflops = pd.concat(all_gflops_series).mean() if all_gflops_series else 0.8
    
    all_times_series = []
    time_adapters_available = ["PED_DENSITY", "ROAD_PRESENCE"]
    for adapter in time_adapters_available:
        time_col = f"pred_{adapter.lower()}_inference_time_ms"
        if time_col in scene_df.columns:
            all_times_series.append(scene_df[time_col].dropna())

    mean_time = pd.concat(all_times_series).mean() if all_times_series else 10.0
    
    defaults = {'time': mean_time, 'gflops': mean_gflops}

    print(f"   - Default GFLOPs set to: {defaults['gflops']:.2f} (average of {len(all_gflops_series)} adapter columns)")
    print(f"   - Default inference time set to: {defaults['time']:.2f} ms (average of {len(all_times_series)} adapter columns)")
    return defaults

# =============================================================================
# C: PERFORMANCE SIMULATION
# =============================================================================

def simulate_approach_performance(scene_df: pd.DataFrame, mode: str, defaults: Dict) -> Dict:
    """
    Simulates the performance of the 'ALL' vs. 'ADAPTIVE' adapter firing strategies.
    
    Args:
        scene_df: DataFrame containing the scene context data.
        mode: The simulation mode ('all' or 'adaptive').
        defaults: A dictionary with fallback values for 'time' and 'gflops'.

    Returns:
        A dictionary of aggregated performance metrics for the specified mode.
    """

    BEHAVIOR_ADAPTERS = ["ACTION", "LOOK", "CROSS", "OCCLUSION"]
    SCENE_ADAPTERS = ["WEATHER", "TIME_OF_DAY", "PED_DENSITY", "ROAD_PRESENCE"]
    ALL_ADAPTERS = BEHAVIOR_ADAPTERS + SCENE_ADAPTERS
    
    adapter_time_cols = {a: f"pred_{a.lower()}_inference_time_ms" for a in ALL_ADAPTERS}
    adapter_gflops_cols = {a: f"gflops_{a.lower()}" for a in ALL_ADAPTERS}

    adapter_usage = {a: 0 for a in ALL_ADAPTERS}
    total_frames = len(scene_df)
    total_adapters_fired = 0
    per_frame_times = np.zeros(total_frames)
    per_frame_gflops = np.zeros(total_frames)
    per_frame_adapters = np.zeros(total_frames)

    def get_adapter_time(row, adapter):
        col = adapter_time_cols.get(adapter, "")
        return row.get(col, defaults['time'])

    def get_adapter_gflops(row, adapter):
        col = adapter_gflops_cols.get(adapter, "")
        return row.get(col, defaults['gflops'])

    last_video_id = None
    last_ped_density_result = None
    video_ped_density_result = {}

    for i, (_, row) in enumerate(scene_df.iterrows()):
        video_id = row.get("video_id")
        frame_id = row.get("frame_id")
        is_first_frame = (video_id != last_video_id)
        if is_first_frame:
            last_ped_density_result = None
            video_ped_density_result[video_id] = None
        last_video_id = video_id

        adapters_fired_this_frame = []
        
        # --- ALL mode ---
        if mode == "all":
            # Fire all BEHAVIOR_ADAPTERS every frame
            adapters_fired_this_frame = BEHAVIOR_ADAPTERS.copy()
            # Fire all SCENE_ADAPTERS every 30th frame
            if frame_id % 30 == 0:
                adapters_fired_this_frame += SCENE_ADAPTERS
        
        # --- ADAPTIVE mode ---
        else:
            # 1. WEATHER and TIME_OF_DAY only on first frame
            if is_first_frame:
                adapters_fired_this_frame += ["WEATHER", "TIME_OF_DAY"]
            # 2. PED_DENSITY every 30th frame
            if frame_id % 30 == 0:
                adapters_fired_this_frame.append("PED_DENSITY")
                ped_density_result = row.get("pred_ped_density", "unknown")
                last_ped_density_result = ped_density_result
                video_ped_density_result[video_id] = ped_density_result
            else:
                ped_density_result = video_ped_density_result.get(video_id, None)
            # 3. Get scene context
            weather = row.get("pred_weather", "unknown").lower()
            time_of_day = row.get("pred_time_of_day", "unknown").lower()
            # 4. Adapter firing logic
            if (time_of_day == "night" or weather in ["rainy", "snowy"] or (last_ped_density_result == "low_pedestrian_density")):
                adapters_fired_this_frame += ["ACTION", "LOOK", "CROSS", "OCCLUSION"]
            elif last_ped_density_result == "medium_pedestrian_density":
                adapters_fired_this_frame += ["ACTION", "LOOK", "CROSS", "ROAD_PRESENCE"]
            elif last_ped_density_result == "high_pedestrian_density" and not (time_of_day == "night" or weather in ["rainy", "snowy"]):
                adapters_fired_this_frame += ["ROAD_PRESENCE"]
            # else: no additional adapters

        adapters_fired_this_frame = list(set(adapters_fired_this_frame))
        frame_time = sum(get_adapter_time(row, a) for a in adapters_fired_this_frame)
        frame_gflops = sum(get_adapter_gflops(row, a) for a in adapters_fired_this_frame)

        total_adapters_fired += len(adapters_fired_this_frame)
        for adapter in adapters_fired_this_frame:
            adapter_usage[adapter] += 1
        
        per_frame_times[i] = frame_time
        per_frame_gflops[i] = frame_gflops
        per_frame_adapters[i] = len(adapters_fired_this_frame)

    avg_fps = 1000 / np.mean(per_frame_times) if np.mean(per_frame_times) > 0 else 0
    return {
        "mode": mode.upper(),
        "avg_inference_time_ms": np.mean(per_frame_times),
        "avg_fps": avg_fps,
        "avg_gflops": np.mean(per_frame_gflops),
        "avg_adapters_fired_per_frame": np.mean(per_frame_adapters),
        "total_adapters_fired": total_adapters_fired,
        "avg_time_per_active_adapter": np.sum(per_frame_times) / total_adapters_fired if total_adapters_fired > 0 else 0,
        "adapter_utilization": {a: count / total_frames for a, count in adapter_usage.items()}
    }

# =============================================================================
# D: VISUALIZATION & REPORTING
# =============================================================================

def plot_ablation_results(all_metrics: Dict, adaptive_metrics: Dict):
    """Creates and saves a grouped bar plot comparing key performance metrics."""

    import numpy as np
    import matplotlib.colors as mcolors

    metrics = [
        ("Avg Inference Time (ms)", "avg_inference_time_ms"),
        ("Avg FPS", "avg_fps"),
        ("Avg GFLOPs", "avg_gflops"),
        ("Avg Adapters Fired / Frame", "avg_adapters_fired_per_frame"),
    ]
    all_vals = [all_metrics[k] for _, k in metrics]
    adaptive_vals = [adaptive_metrics[k] for _, k in metrics]
    percent_change = [
        100 * (adaptive - all_val) / all_val if all_val != 0 else 0
        for all_val, adaptive in zip(all_vals, adaptive_vals)
    ]

    base_colors = COLORS[:len(metrics)]
    def lighten(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        white = np.array([1, 1, 1])
        return tuple(c + (white - c) * amount)
    light_colors = [lighten(c, 0.5) for c in base_colors]

    x = np.arange(len(metrics)) * 0.7  
    width = 0.18  
    offset = width / 2.5  

    xlabels = []
    for i, (name, _) in enumerate(metrics):
        pct = percent_change[i]
        sign = "+" if pct >= 0 else ""
        color = "green" if pct >= 0 else "red"
        pct_str = f"{{\\color{{{color}}}{sign}{pct:.1f}\\%}}"
        xlabels.append(f"{name}\n" + r"$\mathdefault{" + f"{sign}{pct:.1f}\\%" + r"}$")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = []
    bars2 = []
    for i in range(len(metrics)):
        b1 = ax.bar(x[i] - offset, all_vals[i], width, color=base_colors[i], label=f"{metrics[i][0]} (ALL)" if i == 0 else "")
        b2 = ax.bar(x[i] + offset, adaptive_vals[i], width, color=light_colors[i], label=f"{metrics[i][0]} (ADAPTIVE)" if i == 0 else "")
        bars1.append(b1)
        bars2.append(b2)

    for i, b1 in enumerate(bars1):
        height = b1[0].get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(b1[0].get_x() + b1[0].get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=base_colors[i])
    for i, b2 in enumerate(bars2):
        height = b2[0].get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(b2[0].get_x() + b2[0].get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=base_colors[i])

    ax.set_ylabel('Value', fontsize=13)
    ax.set_xticks(x)
    for label, pct in zip(ax.set_xticklabels(xlabels, fontsize=12, rotation=35, ha='right'), percent_change):
        sign = "+" if pct >= 0 else ""
        color = "green" if pct >= 0 else "red"
        label.set_color('black')
        label.set_fontweight('bold')

    ax.set_title('Ablation Study: ALL vs. ADAPTIVE (All Metrics)', fontsize=15, fontweight='bold')

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=base_colors[0], label='ALL'),
        Patch(facecolor=light_colors[0], label='ADAPTIVE')
    ]
    ax.legend(handles=legend_patches, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    
    plt.savefig(ABLATION_DIR / "adaptive_vs_all_ablation.png", dpi=300)
    plt.close()
    print(f"\nVisualization saved to '{ABLATION_DIR / 'adaptive_vs_all_ablation.png'}'")

def save_ablation_report(all_metrics: Dict, adaptive_metrics: Dict):
    """Saves a detailed markdown report comparing the two operational modes."""
    with open(ABLATION_DIR / "ablation_report.md", "w") as f:
        f.write("# Ablation Study: Adaptive vs. All Adapters\n\n")
        f.write("This report compares the performance of the system when firing all adapters on every frame versus using an adaptive, rule-based approach.\n\n")
        f.write("| Metric | ALL Mode | ADAPTIVE Mode | Change |\n")
        f.write("|---|---|---|---|\n")
        def format_row(metric, key, higher_is_better=True, unit=""):
            all_val = all_metrics[key]
            adaptive_val = adaptive_metrics[key]
            if all_val == 0:
                change = float('inf') if adaptive_val > 0 else 0
            else:
                change = ((adaptive_val - all_val) / all_val) * 100
            emoji = "✅" if (change > 0 and higher_is_better) or (change < 0 and not higher_is_better) else "❌"
            return f"| **{metric}** | {all_val:.2f}{unit} | {adaptive_val:.2f}{unit} | {change:+.1f}% {emoji} |\n"
        f.write(format_row("Avg Inference Time", "avg_inference_time_ms", higher_is_better=False, unit=" ms"))
        f.write(format_row("Avg FPS", "avg_fps", higher_is_better=True))
        f.write(format_row("Avg GFLOPs", "avg_gflops", higher_is_better=False))
        f.write(format_row("Avg Adapters Fired", "avg_adapters_fired_per_frame", higher_is_better=False))
        f.write(format_row("Avg Time / Active Adapter", "avg_time_per_active_adapter", higher_is_better=False, unit=" ms"))
        f.write("\n### Adapter Utilization (% of frames)\n\n")
        f.write("| Adapter | ALL Mode | ADAPTIVE Mode |\n")
        f.write("|---|---|---|\n")
        for adapter in all_metrics["adapter_utilization"]:
            all_util = all_metrics["adapter_utilization"][adapter] * 100
            adaptive_util = adaptive_metrics["adapter_utilization"][adapter] * 100
            f.write(f"| {adapter} | {all_util:.1f}% | {adaptive_util:.1f}% |\n")
            
    print(f"Markdown report saved to '{ABLATION_DIR / 'ablation_report.md'}'")

# =============================================================================
# D. MAIN EXECUTION FLOW
# =============================================================================
def main(scene_csv: str, behavior_csv: Optional[str] = None) -> None:
    """Main execution function to run the complete ablation study."""
    print("=" * 60)
    print(" Starting Ablation Study: ALL (Baseline) vs. ADAPTIVE")
    print("=" * 60)

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    
    scene_df, behavior_df = load_ablation_data(scene_csv, behavior_csv)
    default_metrics = calculate_default_metrics(scene_df, behavior_df)

    print("\n[1/2] Simulating 'ALL' mode (baseline)...")
    all_metrics = simulate_approach_performance(scene_df, mode="all", defaults=default_metrics)
    print("   ... 'ALL' mode simulation complete.")

    print("\n[2/2] Simulating 'ADAPTIVE' mode...")
    adaptive_metrics = simulate_approach_performance(scene_df, mode="adaptive", defaults=default_metrics)
    print("   ... 'ADAPTIVE' mode simulation complete.")

    print("\nGenerating report and visualizations...")
    plot_ablation_results(all_metrics, adaptive_metrics)
    save_ablation_report(all_metrics, adaptive_metrics)

    print("\n" + "=" * 60)
    print(" Ablation Study Successfully Completed!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an ablation study comparing a baseline 'ALL' adapter strategy with an 'ADAPTIVE' strategy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--scene_csv",
        type=str,
        default=os.path.join("Generated_Data", "raw_data_scene_contextual_analysis_step2.csv"),
        help="Path to the scene contextual analysis CSV file from Step 2."
    )
    parser.add_argument(
        "--behavior_csv",
        type=str,
        default=os.path.join("Generated_Data", "raw_data_pedestrian_behavior_step2.csv"),
        help="Path to the pedestrian behavior analysis CSV file from Step 2 (optional)."
    )
    args = parser.parse_args()
    main(args.scene_csv, args.behavior_csv)
