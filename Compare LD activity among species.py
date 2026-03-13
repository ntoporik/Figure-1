"""
Species Comparison for LD Activity Data with Circular Statistics
=================================================================
MODIFIED VERSION - Uses comprehensive analysis CSV files for consistency with Figure 2

This script compares locomotor activity across three spider species under LD conditions
and generates circular plots showing phase relationships.

METHODOLOGY (consistent with Figure 2):
- Significance: Period_p_value < 0.05 (from Lomb-Scargle analysis)
- Entrainment: Period within 5% of 24h (22.8-25.2 hours)

INSTRUCTIONS:
1. Paths are pre-configured below
2. Run the script in Spyder (F5)
3. Outputs will be saved in the same directory as this script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# ============================================================================
# CONFIGURATION: Paths to comprehensive analysis CSV files
# ============================================================================

DATA_FILES = {
    'Larinioides': r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Larenioides\LC_spider_analysis_comprehensive_with_LD_split.csv",
    'Agelenopsis': r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Agelenopsis\Ag_spider_analysis_comprehensive_with_LD_split.csv",
    'Steatoda':    r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Steatoda\Sg_spider_analysis_comprehensive_with_LD_split.csv"
}

# Original raw data folders (for daily activity profiles)
AGELENOPSIS_FOLDER = r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Agelenopsis"
LARINIOIDES_FOLDER = r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Larenioides"
STEATODA_FOLDER    = r"C:\Users\toporikovan\Box\students research\Stochastic Spiders\Steatoda Paper\Comparative actibity analysis\Data analysis\Steatoda"

# Output filenames
OUTPUT_CSV = 'species_daily_activity_comparison_LD_v2.csv'
OUTPUT_FIGURE = 'species_comparison_LD_with_circular_v2.png'

# Analysis parameters (consistent with Figure 2)
SIGNIFICANCE_ALPHA = 0.05  # p < 0.05 for significant periodicity
ENTRAINMENT_TOL = 0.05     # 5% around 24h (22.8-25.2 hours)

# ============================================================================
# HELPER FUNCTIONS FOR RAW DATA PROCESSING (for activity profiles)
# ============================================================================

def is_locomotor_monitor_file(df):
    """Check if dataframe is a locomotor monitor activity file"""
    columns = df.columns.tolist()
    has_datetime = any(col.lower() == 'datetime' for col in columns)
    has_light = any(col.lower() == 'light' for col in columns)
    
    if not has_datetime or not has_light:
        return False
    
    total_cols = len(columns)
    activity_cols = total_cols - 2
    
    if activity_cols < 1 or activity_cols > 32:
        return False
    
    return True


def find_ld_files_in_folder(folder_path):
    """Find all locomotor monitor CSV files containing 'LD'"""
    ld_files = []
    
    if not os.path.exists(folder_path):
        print(f"WARNING: Folder not found: {folder_path}")
        return ld_files
    
    for file in os.listdir(folder_path):
        if 'LD' in file and file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            
            try:
                df = pd.read_csv(file_path, nrows=5)
                if is_locomotor_monitor_file(df):
                    ld_files.append(file_path)
            except Exception as e:
                pass
    
    return ld_files


def find_zt0(light_series):
    """Find ZT0 (lights-on time) from Light column"""
    for i in range(1, len(light_series)):
        if light_series.iloc[i-1] == 0 and light_series.iloc[i] == 1:
            return i
    
    if light_series.iloc[0] == 1:
        return 0
    
    for i in range(len(light_series)):
        if light_series.iloc[i] == 1:
            return i
    
    return 0


def process_locomotor_file_for_daily_avg(file_path):
    """Process file for daily average activity"""
    try:
        df = pd.read_csv(file_path)
        
        datetime_col = [col for col in df.columns if col.lower() == 'datetime'][0]
        light_col = [col for col in df.columns if col.lower() == 'light'][0]
        activity_cols = [col for col in df.columns 
                        if col.lower() not in ['datetime', 'light']]
        
        if not activity_cols:
            return None, None
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        zt0_index = find_zt0(df[light_col])
        df['minutes_since_start'] = range(len(df))
        df['ZT'] = (df['minutes_since_start'] - zt0_index) % 1440
        
        spider_daily_averages = {}
        for spider_col in activity_cols:
            daily_avg = df.groupby('ZT')[spider_col].mean()
            spider_daily_averages[spider_col] = daily_avg
        
        light_daily = df.groupby('ZT')[light_col].first()
        daily_df = pd.DataFrame(spider_daily_averages).sort_index()
        light_daily = light_daily.sort_index()
        
        return daily_df, light_daily
        
    except Exception as e:
        return None, None


def load_and_average_species_data(files_list, species_name):
    """Load and average activity data from multiple files"""
    all_activity = []
    
    for file_path in files_list:
        daily_df, light_status = process_locomotor_file_for_daily_avg(file_path)
        if daily_df is not None:
            for col in daily_df.columns:
                all_activity.append(daily_df[col].values)
    
    if not all_activity:
        return None, None
    
    avg_activity = np.mean(all_activity, axis=0)
    return avg_activity, light_status


def get_dark_periods(light_status, x_values):
    """Identify dark periods for shading"""
    dark_periods = []
    in_dark = False
    start_x = None
    
    for i, (x, light) in enumerate(zip(x_values, light_status)):
        if light == 0 and not in_dark:
            start_x = x
            in_dark = True
        elif light == 1 and in_dark:
            dark_periods.append((start_x, x))
            in_dark = False
    
    if in_dark:
        dark_periods.append((start_x, x_values.max()))
    
    return dark_periods

# ============================================================================
# CIRCULAR STATISTICS FUNCTIONS
# ============================================================================

def calc_circular_stats_from_file(file_path, spider_id):
    """Calculate circular statistics for one spider from raw activity file"""
    try:
        df = pd.read_csv(file_path)
        
        datetime_col = [col for col in df.columns if col.lower() == 'datetime'][0]
        light_col = [col for col in df.columns if col.lower() == 'light'][0]
        spider_cols = [col for col in df.columns 
                      if col.lower() not in ['datetime', 'light']]
        
        if spider_id not in spider_cols:
            return None
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        zt0_index = find_zt0(df[light_col])
        df['ZT'] = (np.arange(len(df)) - zt0_index) % 1440 / 60
        
        activity = df[spider_id].values
        zt = df['ZT'].values
        
        active = activity > 0
        if np.sum(active) < 10:
            return None
        
        weighted = np.repeat(zt[active], activity[active].astype(int))
        if len(weighted) == 0:
            return None
        
        rad = (weighted % 24) / 24 * 2 * np.pi
        n = len(rad)
        x, y = np.sum(np.cos(rad)), np.sum(np.sin(rad))
        
        r = np.sqrt(x**2 + y**2) / n
        theta = np.arctan2(y, x)
        phase = (theta if theta >= 0 else theta + 2*np.pi) / (2*np.pi) * 24
        
        return {'mean_phase_ZT': phase, 'vector_strength': r}
        
    except Exception as e:
        return None

# ============================================================================
# LOAD COMPREHENSIVE DATA AND CALCULATE CIRCULAR STATS
# ============================================================================

def load_species_comprehensive_data(csv_path, raw_folder, species_name):
    """
    Load comprehensive CSV and calculate circular statistics
    
    Returns:
        circular_df: DataFrame with period, phase, and classification info
        daily_activity: Average daily activity profile
        light_status: Light status for plotting
    """
    # Load comprehensive analysis
    df = pd.read_csv(csv_path)
    
    # Filter for LD condition only
    ld_df = df[df['Condition'] == 'LD'].copy()
    
    # Apply significance and entrainment criteria (same as Figure 2)
    ld_df = ld_df.dropna(subset=['Period_hours'])
    ld_df['is_significant'] = ld_df['Period_p_value'] < SIGNIFICANCE_ALPHA
    ld_df['is_entrained'] = False
    
    for idx, row in ld_df.iterrows():
        if not pd.isna(row['Period_hours']):
            period = row['Period_hours']
            is_ent = 24*(1-ENTRAINMENT_TOL) <= period <= 24*(1+ENTRAINMENT_TOL)
            ld_df.at[idx, 'is_entrained'] = is_ent
    
    # Get daily activity profiles from raw files
    raw_files = find_ld_files_in_folder(raw_folder)
    daily_activity, light_status = load_and_average_species_data(raw_files, species_name)
    
    # Calculate circular statistics for each spider
    circular_data = []
    
    for idx, row in ld_df.iterrows():
        spider_id = row['Spider_ID']
        
        # Try to find the spider in raw files and calculate circular stats
        circ_stats = None
        for file_path in raw_files:
            circ_stats = calc_circular_stats_from_file(file_path, spider_id)
            if circ_stats is not None:
                break
        
        if circ_stats is not None:
            circular_data.append({
                'spider_id': spider_id,
                'period_hr': row['Period_hours'],
                'period_p_value': row['Period_p_value'],
                'mean_phase_ZT': circ_stats['mean_phase_ZT'],
                'vector_strength': circ_stats['vector_strength'],
                'is_significant': row['is_significant'],
                'is_entrained': row['is_entrained'],
                'total_activity': row['Total_Crossings']
            })
    
    circular_df = pd.DataFrame(circular_data)
    
    return circular_df, daily_activity, light_status

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def setup_polar_axis(ax):
    """Configure polar axis for circular plots"""
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('S')
    ax.set_ylim(0, 1)
    
    ticks = np.arange(0, 24, 6)
    ax.set_thetagrids(ticks * 15, [f'{int(z)}' for z in ticks], fontsize=10)
    
    dark = np.linspace(np.pi, 2*np.pi, 100)
    ax.fill_between(dark, 0, 1, alpha=0.3, color='lightblue', zorder=1)
    
    ax.grid(True, alpha=0.3)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8, alpha=0.7)


def calc_mean_vector(thetas):
    """Calculate population mean vector"""
    x = np.mean(np.cos(thetas))
    y = np.mean(np.sin(thetas))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (theta if theta >= 0 else theta + 2*np.pi), r


def create_combined_figure(daily_data, circular_data, light_status, output_file):
    """Create 2x3 figure with activity and circular plots"""
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    
    fig = plt.figure(figsize=(12, 8))
    
    species_config = [
        ('Larinioides', 'green'),
        ('Agelenopsis', 'red'),
        ('Steatoda', 'black')
    ]
    
    # Find global max for activity plots
    global_max = 0
    for species, _ in species_config:
        if species in daily_data.columns:
            global_max = max(global_max, daily_data[species].max())
    
    # Create plots for each species
    for i, (species, color) in enumerate(species_config):
        # Top row: Activity plots
        ax_top = plt.subplot(2, 3, i+1)
        
        if species in daily_data.columns:
            x_values = daily_data['time_hours'] + daily_data['time_minutes']/60
            dark_periods = get_dark_periods(light_status[species], x_values)
            
            for start, end in dark_periods:
                ax_top.axvspan(start, end, alpha=0.3, color='lightblue', zorder=1)
            
            ax_top.bar(x_values, daily_data[species], color=color, width=0.02)
            ax_top.set_ylim(0, global_max * 1.05)
        
        ax_top.set_xlabel('ZT (hours)', fontsize=11)
        if i == 0:
            ax_top.set_ylabel('Activity (crossings/minute)', fontsize=11)
        
        ax_top.grid(True, alpha=0.3, linewidth=0.5)
        ax_top.set_xlim(0, 24)
        ax_top.set_xticks(range(0, 25, 6))
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        # Bottom row: Circular plots
        ax_bottom = plt.subplot(2, 3, i+4, projection='polar')
        
        if species in circular_data:
            df = circular_data[species]
            df = df.dropna(subset=['mean_phase_ZT', 'vector_strength'])
            df = df[df['vector_strength'] > 0].copy()
            
            if len(df) > 0:
                df['theta'] = df['mean_phase_ZT'] / 24 * 2 * np.pi
                
                non_sig = df[~df['is_significant']]
                sig_not_ent = df[df['is_significant'] & ~df['is_entrained']]
                entrained = df[df['is_significant'] & df['is_entrained']]
                
                if len(non_sig) > 0:
                    ax_bottom.scatter(non_sig['theta'], non_sig['vector_strength'],
                                    facecolors='none', edgecolors=color,
                                    s=80, linewidth=1.5, zorder=2)
                
                if len(sig_not_ent) > 0:
                    ax_bottom.scatter(sig_not_ent['theta'], sig_not_ent['vector_strength'],
                                    c=color, s=80, alpha=0.3,
                                    edgecolors=color, linewidth=1, zorder=3)
                
                if len(entrained) > 0:
                    ax_bottom.scatter(entrained['theta'], entrained['vector_strength'],
                                    c=color, s=80, alpha=0.9,
                                    edgecolors=color, linewidth=2.5, zorder=4)
                    
                    mean_theta, mean_r = calc_mean_vector(entrained['theta'].values)
                    ax_bottom.annotate('', xy=(mean_theta, mean_r), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', 
                                                   color='darkgray' if color == 'black' else 'dark'+color,
                                                   lw=3),
                                     zorder=5)
        
        setup_polar_axis(ax_bottom)
        if i == 0:
            ax_bottom.set_ylabel('Vector Strength (r)', fontsize=11, labelpad=30)
    
    # Add single legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='gray', markersize=8, linewidth=2.5,
               label='Entrained'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='gray', markersize=8, alpha=0.3, linewidth=1,
               label='Non-entrained'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=8, linewidth=1.5,
               label='Non-significant')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=3, frameon=False, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Figure saved: {output_file}")


# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("="*70)
print("Species Comparison - LD Activity (v2 - Consistent Methodology)")
print("="*70)

species_folders = {
    'Agelenopsis': AGELENOPSIS_FOLDER,
    'Larinioides': LARINIOIDES_FOLDER,
    'Steatoda': STEATODA_FOLDER
}

# Process each species
daily_activity = {}
circular_stats = {}
light_statuses = {}

for species in ['Larinioides', 'Agelenopsis', 'Steatoda']:
    print(f"\nProcessing {species}...")
    
    csv_path = DATA_FILES[species]
    raw_folder = species_folders[species]
    
    circ_df, avg_activity, light_status = load_species_comprehensive_data(
        csv_path, raw_folder, species
    )
    
    if avg_activity is not None:
        daily_activity[species] = avg_activity
        light_statuses[species] = light_status
        print(f"  Daily activity: OK")
    
    if circ_df is not None and len(circ_df) > 0:
        circular_stats[species] = circ_df
        
        total = len(circ_df)
        sig = circ_df['is_significant'].sum()
        ent = (circ_df['is_significant'] & circ_df['is_entrained']).sum()
        
        print(f"  Total spiders: {total}")
        print(f"  Significant (p<0.05): {sig} ({sig/total*100:.1f}%)")
        print(f"  Entrained (sig + 22.8-25.2h): {ent} ({ent/total*100:.1f}%)")
        
        # Save individual species stats
        circ_df.to_csv(f"{species}_circular_stats_v2.csv", index=False)

# Create combined dataframe for daily activity
min_length = min(len(data) for data in daily_activity.values())
result_data = {}
for minute in range(min_length):
    hours = minute // 60
    minutes = minute % 60
    result_data[minute] = {
        'time_hours': hours,
        'time_minutes': minutes,
        'ZT': minute
    }
    for species, activity_data in daily_activity.items():
        result_data[minute][species] = activity_data[minute]

comparison_df = pd.DataFrame.from_dict(result_data, orient='index')

# Save data
comparison_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSUCCESS: Data saved: {OUTPUT_CSV}")

# Create combined figure
print("\nGenerating combined figure...")
create_combined_figure(comparison_df, circular_stats, light_statuses, OUTPUT_FIGURE)

# Print summary
print("\n" + "="*70)
print("Summary Statistics:")
print("="*70)
for species in ['Larinioides', 'Agelenopsis', 'Steatoda']:
    if species in comparison_df.columns:
        print(f"{species:15s}: avg={comparison_df[species].mean():.4f}, "
              f"max={comparison_df[species].max():.4f}")

print("\nAnalysis complete!")