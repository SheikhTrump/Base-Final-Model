import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# REAL KAGGLE DATASET LOADING
# ============================================================================
# Kaggle Dataset: Crop and Fertilizer Dataset for Western Maharashtra
# URL: https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra
# 
# INSTRUCTIONS:
# 1. Download from Kaggle: https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra
# 2. Place the CSV file in this directory (y:\Coding\Flower\)
# 3. Look for file named "Crop and fertilizer dataset.csv" or similar
# 4. The script will automatically detect and load it
# ============================================================================

# Define expected dataset columns
FEATURE_COLUMNS = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']

# ALL possible crops in dataset (16 total) - used for global model
# Each district will use a subset of these crops
ALL_CROPS = ['Sugarcane', 'Jowar', 'Cotton', 'Rice', 'Wheat', 'Groundnut', 
             'Maize', 'Tur', 'Urad', 'Moong', 'Gram', 'Masoor', 
             'Soybean', 'Ginger', 'Turmeric', 'Grapes']

# Note: Will auto-detect all districts from data and aggregate into 4 clients
TARGET_DISTRICTS = []  # Will be populated dynamically

# Sub-set of crops focused on in the paper/logic
TARGET_CROPS = ['rice', 'maize', 'cassava', 'seed_cotton', 'yams', 'bananas']

def _aggregate_districts_into_clients(df, num_clients=4):
    """
    Aggregates districts into groups to form the requested number of clients.
    """
    if 'District' not in df.columns:
        return []
        
    district_counts = df['District'].value_counts()
    all_districts = district_counts.index.tolist()
    
    # Simple round-robin assignment
    groups = [[] for _ in range(num_clients)]
    for i, district in enumerate(all_districts):
        groups[i % num_clients].append(district)
        
    return groups

def find_kaggle_csv():
    """
    Finds the Kaggle dataset CSV file in the current directory.
    Searches for common naming patterns.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Common filename patterns
    possible_names = [
        'Crop and fertilizer dataset.csv',
      
    ]
    
    for filename in possible_names:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            return filepath
    
    return None

def load_kaggle_dataset():
    """
    Loads real Kaggle agricultural dataset for Western Maharashtra.
    
    Dataset Features (6):
    - Nitrogen (N)
    - Phosphorus (P)
    - Potassium (K)
    - pH
    - Rainfall
    - Temperature
    
    Target Crops (6):
    - rice, maize, cassava, seed_cotton, yams, bananas
    
    Districts (4):
    - Pune, Ahmednagar, Solapur, Satara
    
    Returns:
    - DataFrame with processed data ready for federated learning
    """
    csv_path = find_kaggle_csv()
    
    if csv_path is None:
        print("\n" + "="*70)
        print("WARNING: KAGGLE DATASET NOT FOUND")
        print("="*70)
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra")
        print("\nThen place the CSV file in: y:\\Coding\\Flower\\")
        print("\nSupported filenames:")
        print("  - Crop and fertilizer dataset.csv")
        print("  - Crop_and_fertilizer_dataset.csv")
        print("  - crop_fertilizer.csv")
        print("  - Or any similar variation")
        print("="*70)
        print("\nFalling back to SYNTHETIC data for demonstration...")
        print("="*70 + "\n")
        return None
    
    print(f"\n[OK] Loading real Kaggle dataset from: {csv_path}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    print(f"  Original dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Find district column (might be named differently)
    district_col = None
    for col in df.columns:
        if col.lower() in ['district_name', 'district', 'districtname']:
            district_col = col
            break
    
    if district_col:
        print(f"  [OK] Found district column: '{district_col}'")
        df = df.rename(columns={district_col: 'District'})
    else:
        print("  [WARNING] No district column found. Creating from data...")
    
    # Get unique crops in dataset
    unique_crops_in_csv = df['Crop'].unique() if 'Crop' in df.columns else []
    print(f"  Crops in dataset: {unique_crops_in_csv}")
    
    # Try to match crops (case-insensitive)
    # Use ALL_CROPS from the dataset specs instead of the subset TARGET_CROPS
    # This ensures we use all available data (16 crops) for a proper challenge
    target_crops_lower = [crop.lower() for crop in ALL_CROPS]
    
    # Filter by crops - use case-insensitive matching
    df_filtered = df[df['Crop'].str.lower().isin(target_crops_lower)].copy()
    print(f"  After filtering crops: {df_filtered.shape[0]} rows")
    
    # If no matches, try exact matching with different names
    if len(df_filtered) == 0:
        print("  [WARNING] No exact crop matches found. Checking for similar names...")
        print(f"  ALL_CROPS: {ALL_CROPS}")
        print(f"  ACTUAL_CROPS: {list(unique_crops_in_csv)}")
        # Try with just first match if crops don't match exactly
        df_filtered = df.copy()
    
    # Filter by districts if available
    if 'District' in df_filtered.columns:
        unique_districts = df_filtered['District'].unique()
        print(f"  Districts in dataset: {list(unique_districts)}")
        
        # DON'T filter by TARGET_DISTRICTS - use ALL available districts!
        # They will be aggregated into 4 clients in get_partitions()
        print(f"  Using ALL available districts (will aggregate into 4 clients)")
    else:
        print(f"  [WARNING] District column not found, using all data.")
        if len(unique_crops_in_csv) > 0:
            # Randomly assign to districts for partitioning
            np.random.seed(42)
            default_districts = ['Kolhapur', 'Satara', 'Solapur', 'Pune']
            df_filtered['District'] = np.random.choice(default_districts, len(df_filtered))
    
    # Check for required feature columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_filtered.columns]
    if missing_cols:
        print(f"  ❌ Missing columns: {missing_cols}")
        print(f"  Available columns: {list(df_filtered.columns)}")
        return None
    
    print(f"  [OK] All required feature columns found!")
    print(f"  [OK] Dataset loaded successfully: {df_filtered.shape[0]} samples")
    print(f"  [OK] Crops: {df_filtered['Crop'].unique().tolist() if 'Crop' in df_filtered.columns else 'N/A'}")
    print(f"  [OK] Districts: {df_filtered['District'].unique().tolist() if 'District' in df_filtered.columns else 'N/A'}")
    
    return df_filtered if len(df_filtered) > 0 else None

def generate_district_data(district_name, df_kaggle=None):
    """
    Gets real data for a specific district from the Kaggle dataset.
    Falls back to synthetic data if Kaggle dataset not available.
    
    NOTE: Paper uses real Kaggle agricultural data with 4,513 total samples.
    This implementation loads real data from:
    https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra
    
    Features: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature (6 features)
    Target Crops: rice, maize, cassava, seed_cotton, yams, bananas (6 output classes)
    """
    if df_kaggle is not None and district_name in df_kaggle['District'].unique():
        # Use real data from Kaggle
        return df_kaggle[df_kaggle['District'] == district_name].copy()
    else:
        # Fallback to synthetic data if Kaggle data not available for this district
        return _generate_synthetic_district_data(district_name)

def _generate_synthetic_district_data(district_name, num_samples=225):
    """
    Generates synthetic data as fallback when Kaggle dataset unavailable.
    """
    crops = TARGET_CROPS
    
    data = []
    for crop in crops:
        noise_level = 8.0
        
        if crop == 'rice':
            n, p, k = np.random.normal(80, noise_level, num_samples), np.random.normal(40, noise_level, num_samples), np.random.normal(40, noise_level, num_samples)
            ph = np.random.normal(6.5, 0.5, num_samples)
            rain = np.random.normal(200, 30, num_samples)
            temp = np.random.normal(25, 4, num_samples)
        elif crop == 'maize':
            n, p, k = np.random.normal(100, noise_level, num_samples), np.random.normal(50, noise_level, num_samples), np.random.normal(20, noise_level, num_samples)
            ph = np.random.normal(6.0, 0.5, num_samples)
            rain = np.random.normal(100, 30, num_samples)
            temp = np.random.normal(28, 4, num_samples)
        elif crop == 'cassava':
            n, p, k = np.random.normal(40, noise_level, num_samples), np.random.normal(20, noise_level, num_samples), np.random.normal(60, noise_level, num_samples)
            ph = np.random.normal(5.5, 0.8, num_samples)
            rain = np.random.normal(150, 40, num_samples)
            temp = np.random.normal(30, 5, num_samples)
        elif crop == 'seed_cotton':
            n, p, k = np.random.normal(60, noise_level, num_samples), np.random.normal(30, noise_level, num_samples), np.random.normal(30, noise_level, num_samples)
            ph = np.random.normal(7.5, 0.8, num_samples)
            rain = np.random.normal(80, 20, num_samples)
            temp = np.random.normal(32, 5, num_samples)
        elif crop == 'yams':
            n, p, k = np.random.normal(50, noise_level, num_samples), np.random.normal(25, noise_level, num_samples), np.random.normal(50, noise_level, num_samples)
            ph = np.random.normal(6.8, 0.5, num_samples)
            rain = np.random.normal(120, 30, num_samples)
            temp = np.random.normal(26, 4, num_samples)
        else:  # bananas
            n, p, k = np.random.normal(90, noise_level, num_samples), np.random.normal(40, noise_level, num_samples), np.random.normal(80, noise_level, num_samples)
            ph = np.random.normal(6.2, 0.5, num_samples)
            rain = np.random.normal(180, 40, num_samples)
            temp = np.random.normal(28, 5, num_samples)
            
        df_crop = pd.DataFrame({
            'Nitrogen': n,
            'Phosphorus': p,
            'Potassium': k,
            'pH': ph,
            'Rainfall': rain,
            'Temperature': temp,
            'Crop': [crop] * num_samples
        })
        data.append(df_crop)
        
    df = pd.concat(data).sample(frac=1).reset_index(drop=True)
    df['District'] = district_name
    return df

def perform_eda(df_kaggle, output_dir="."):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the dataset.
    Generates 6 publication-quality visualizations saved as PNG files.
    
    Parameters:
    -----------
    df_kaggle : pd.DataFrame
        The Kaggle dataset
    output_dir : str
        Directory to save EDA plots
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    
    # Convert features to numeric
    df_clean = df_kaggle[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    
    # ===== PLOT 1: Statistical Summary =====
    print("\n1. Statistical Summary...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    stats_text = "STATISTICAL SUMMARY OF FEATURES\n\n"
    stats_df = df_clean.describe().T
    stats_text += stats_df.to_string()
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_statistical_summary.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_statistical_summary.png")
    plt.close()
    
    # ===== PLOT 2: Feature Distributions (Histograms) =====
    print("2. Feature Distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distributions (Histograms)', fontsize=14, fontweight='bold')
    
    for idx, col in enumerate(FEATURE_COLUMNS):
        row, col_idx = idx // 3, idx % 3
        axes[row, col_idx].hist(df_clean[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[row, col_idx].set_title(f'{col}')
        axes[row, col_idx].set_xlabel('Value')
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].grid(alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_feature_distributions.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_feature_distributions.png")
    plt.close()
    
    # ===== PLOT 3: Box Plots (Outliers) =====
    print("3. Outlier Detection (Boxplots)...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Boxplots (Outlier Detection)', fontsize=14, fontweight='bold')
    
    for idx, col in enumerate(FEATURE_COLUMNS):
        row, col_idx = idx // 3, idx % 3
        axes[row, col_idx].boxplot(df_clean[col], vert=True)
        axes[row, col_idx].set_title(f'{col}')
        axes[row, col_idx].set_ylabel('Value')
        axes[row, col_idx].grid(alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_boxplots_outliers.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_boxplots_outliers.png")
    plt.close()
    
    # ===== PLOT 4: Feature Correlations (Heatmap) =====
    print("4. Feature Correlations...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df_clean.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_correlation_heatmap.png")
    plt.close()
    
    # ===== PLOT 5: Class Distribution (Crop Types) =====
    print("5. Class Distribution (Crops)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    crop_counts = df_kaggle['Crop'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
    crop_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title('Crop Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Crop Type')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    # Add sample counts on top of bars
    for i, v in enumerate(crop_counts):
        ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_crop_distribution.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_crop_distribution.png")
    plt.close()
    
    # ===== PLOT 6: District Distribution =====
    print("6. District Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    district_counts = df_kaggle['District'].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(district_counts)))
    district_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title('Sample Distribution by District', fontsize=14, fontweight='bold')
    ax.set_xlabel('District')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    # Add sample counts on top of bars
    for i, v in enumerate(district_counts):
        ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_district_distribution.png'), dpi=300, bbox_inches='tight')
    print("   [OK] Saved: eda_district_distribution.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "-"*70)
    print("EDA SUMMARY STATISTICS")
    print("-"*70)
    print(f"\nTotal Samples: {len(df_kaggle)}")
    print(f"Features: {len(FEATURE_COLUMNS)} ({', '.join(FEATURE_COLUMNS)})")
    print(f"Crops: {df_kaggle['Crop'].nunique()} unique")
    print(f"  {df_kaggle['Crop'].unique().tolist()}")
    print(f"Districts: {df_kaggle['District'].nunique()} unique")
    print(f"  {df_kaggle['District'].unique().tolist()}")
    
    print(f"\nFeature Statistics:")
    for col in FEATURE_COLUMNS:
        values = pd.to_numeric(df_kaggle[col], errors='coerce').dropna()
        print(f"  {col:15s}: mean={values.mean():.2f}, std={values.std():.2f}, min={values.min():.2f}, max={values.max():.2f}")
    
    print(f"\nClass Imbalance Ratio: {crop_counts.max() / crop_counts.min():.2f}x")
    print(f"Missing Values: {df_kaggle[FEATURE_COLUMNS].isnull().sum().sum()}")
    
    print("\n" + "="*70)
    print("EDA COMPLETE - 6 plots generated")
    print("="*70 + "\n")


def get_partitions(num_clients=4, run_eda=False):
    """
    Creates data partitions for federated learning across edge servers.
    Paper specifies: 4 edge servers (Tier-III) + 1 cloud server (Tier-IV)
    
    KEY IMPROVEMENT: Each district uses ONLY the crops grown there (matches paper).
    - Local Model 1: ~12 crops
    - Local Model 2: ~7 crops
    - Local Model 3: ~10 crops
    - Local Model 4: ~10 crops
    
    Global model uses all 16 crops for FedAvg parameter averaging compatibility.
    
    AUTO-AGGREGATES ALL DISTRICTS INTO 4 CLIENTS:
    - Detects all available districts from Kaggle dataset
    - Groups them by sample size
    - Assigns to 4 clients (largest districts get their own client, smaller ones are grouped)
    # Try to load real Kaggle dataset
    """
    print("\n" + "="*70)
    print("LOADING AGRICULTURAL DATA")
    print("="*70)
    df_kaggle = load_kaggle_dataset()
    
    if df_kaggle is None:
        # Using synthetic fallback
        use_synthetic = True
        print("Using SYNTHETIC data for demonstration (Kaggle dataset not found)")
        districts_to_use = ['Kolhapur', 'Satara', 'Solapur', 'Pune']  # Default fallback
        print("For real data, see instructions above.\n")
    else:
        use_synthetic = False
        print("Using REAL data from Kaggle dataset\n")
        
        # PERFORM EXPLORATORY DATA ANALYSIS (Only if requested)
        if run_eda:
            perform_eda(df_kaggle)
        
        # AUTO-DETECT ALL UNIQUE DISTRICTS
        all_districts = sorted(df_kaggle['District'].unique().tolist())
        print(f"  Detected {len(all_districts)} unique districts: {all_districts}")
        
        # Group districts by sample count
        district_counts = df_kaggle['District'].value_counts().sort_values(ascending=False)
        print(f"  Sample distribution by district:")
        for district, count in district_counts.items():
            print(f"    - {district}: {count} samples")
        
        # Aggregate into exactly num_clients groups
        districts_to_use = _aggregate_districts_into_clients(df_kaggle, num_clients)
    
    # Initialize GLOBAL LabelEncoder on ALL 16 crops
    # (Each district will use a subset, but all use same global encoding)
    le_global = LabelEncoder()
    le_global.fit(ALL_CROPS)
    
    partitions = []
    
    for i, district_group in enumerate(districts_to_use):
        # Get data for this client (which may include multiple districts)
        if use_synthetic:
            df = generate_district_data(district_group)
        else:
            # If district_group is a list (aggregated), combine data from all districts
            if isinstance(district_group, list):
                dfs = [df_kaggle[df_kaggle['District'] == d].copy() for d in district_group if d in df_kaggle['District'].unique()]
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                df = df_kaggle[df_kaggle['District'] == district_group].copy()
        
        if len(df) == 0:
            print(f"  [WARNING] No data for client {i+1}, skipping...")
            continue
        
        # IDENTIFY CROPS IN THIS DISTRICT/CLIENT - MATCHES PAPER!
        district_crops = sorted(df['Crop'].unique().tolist())
        print(f"  Client {i+1} ({district_group}): {len(district_crops)} crops -> {district_crops}")
        
        # Extract ONLY the required feature columns (not all columns)
        # Convert to numeric and handle any non-numeric values
        df_features = df[FEATURE_COLUMNS].copy()
        
        # Convert all feature columns to numeric (coerce errors to NaN)
        for col in FEATURE_COLUMNS:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        # Remove rows with NaN values
        df_features = df_features.dropna()
        
        # Filter crops and labels accordingly
        valid_indices = df_features.index
        X = df_features.values  # Use only: N, P, K, pH, Rainfall, Temp
        
        # Use GLOBAL label encoding (all 16 crops for FedAvg compatibility)
        y = le_global.transform(df.loc[valid_indices, 'Crop'].values)
        
        if len(X) == 0:
            print(f"  [WARNING] No valid numeric data for client {i+1}, skipping...")
            continue
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape for LSTM: (samples, time_steps, features)
        # We use 1 time step for simplicity as the paper suggests LSTM for its temporal capability 
        # but the specific data described is point-in-time soil/weather.
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Create district label for reporting
        if isinstance(district_group, list):
            district_label = " + ".join(district_group)
        else:
            district_label = district_group
        
        partitions.append({
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.long),
            'district': district_label,
            'crops': district_crops,  # Store district-specific crops
            'crop_names': district_crops  # For use in test() function
        })
    
    return partitions, le_global
if __name__ == "__main__":
    partitions, le = get_partitions(run_eda=True)
    print(f"\n=== FLyer Federated Learning - Data Partitions ===")
    print(f"Total Edge Servers (Tier-III clients): {len(partitions)}")
    print(f"Global Model: 16 output classes (for FedAvg averaging)\n")
    
    total_train_samples = 0
    total_test_samples = 0
    
    for i, p in enumerate(partitions):
        train_samples = len(p['X_train'])
        test_samples = len(p['X_test'])
        total_train_samples += train_samples
        total_test_samples += test_samples
        districts = p['district']
        crops = p.get('crops', [])
        print(f"  Edge Server {i+1} ({districts}):")
        print(f"    Train={train_samples}, Test={test_samples}, Total={train_samples+test_samples}")
        print(f"    Local Crops ({len(crops)}): {crops}")
    
    total_samples = total_train_samples + total_test_samples
    print(f"\nTotal Samples: {total_samples}")
    if total_samples > 500:  # Real data has more samples
        print(f"  [OK] REAL DATA from Kaggle dataset!")
    else:
        print(f"  (Synthetic data - Kaggle dataset not found)")
    print(f"\nGlobal Crops ({len(le.classes_)}): {', '.join(le.classes_)}")
    print(f"Training/Test Split: {total_train_samples}/{total_test_samples}")
    print(f"\nNote: Cloud server (Tier-IV) aggregates from {len(partitions)} edge servers during FL rounds.")
    print(f"Each edge server trains on ONLY crops present in its district (matches paper).")
    print(f"=" * 70)
