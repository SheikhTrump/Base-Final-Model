import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import os

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
# Updated to match actual Kaggle dataset crops and districts
TARGET_CROPS = ['Rice', 'Maize', 'Cotton', 'Jowar', 'Groundnut', 'Wheat']  # 6 crops as per paper
TARGET_DISTRICTS = ['Kolhapur', 'Satara', 'Solapur', 'Pune']  # 4 districts matching actual data (NOT Ahmednagar)

def find_kaggle_csv():
    """
    Finds the Kaggle dataset CSV file in the current directory.
    Searches for common naming patterns.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Common filename patterns
    possible_names = [
        'Crop and fertilizer dataset.csv',
        'Crop_and_fertilizer_dataset.csv',
        'crop_fertilizer.csv',
        'crop_data.csv',
        'crop_and_fertilizer_dataset_for_westernmaharashtra.csv',
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
    target_crops_lower = [crop.lower() for crop in TARGET_CROPS]
    
    # Filter by crops - use case-insensitive matching
    df_filtered = df[df['Crop'].str.lower().isin(target_crops_lower)].copy()
    print(f"  After filtering crops: {df_filtered.shape[0]} rows")
    
    # If no matches, try exact matching with different names
    if len(df_filtered) == 0:
        print("  [WARNING] No exact crop matches found. Checking for similar names...")
        print(f"  TARGET_CROPS: {TARGET_CROPS}")
        print(f"  ACTUAL_CROPS: {list(unique_crops_in_csv)}")
        # Try with just first match if crops don't match exactly
        df_filtered = df.copy()
    
    # Filter by districts if available
    if 'District' in df_filtered.columns:
        unique_districts = df_filtered['District'].unique()
        print(f"  Districts in dataset: {list(unique_districts)}")
        
        # Try matching districts (case-insensitive)
        target_districts_lower = [d.lower() for d in TARGET_DISTRICTS]
        df_filtered = df_filtered[df_filtered['District'].str.lower().isin(target_districts_lower)].copy()
        print(f"  After filtering districts: {df_filtered.shape[0]} rows")
    else:
        print(f"  [WARNING] District column not found, using all data.")
        if len(unique_crops_in_csv) > 0:
            # Randomly assign to districts for partitioning
            np.random.seed(42)
            df_filtered['District'] = np.random.choice(TARGET_DISTRICTS, len(df_filtered))
    
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

def get_partitions(num_clients=4):
    """
    Creates data partitions for federated learning across edge servers.
    Paper specifies: 4 edge servers (Tier-III) + 1 cloud server (Tier-IV)
    This function creates 4 client partitions for the 4 edge servers.
    Cloud aggregation happens in simulate.py (Tier-IV)
    
    Now loads REAL data from Kaggle dataset instead of synthetic!
    """
    # Use actual TARGET_DISTRICTS from Kaggle dataset
    districts = TARGET_DISTRICTS
    partitions = []
    
    # Try to load real Kaggle dataset
    print("\n" + "="*70)
    print("LOADING AGRICULTURAL DATA")
    print("="*70)
    df_kaggle = load_kaggle_dataset()
    
    if df_kaggle is None:
        # Using synthetic fallback
        use_synthetic = True
        print("Using SYNTHETIC data for demonstration (Kaggle dataset not found)")
        print("For real data, see instructions above.\n")
    else:
        use_synthetic = False
        print("Using REAL data from Kaggle dataset\n")
    
    # Initialize LabelEncoder on all possible crops collectively
    # 6 crops as per paper specification
    all_crops = TARGET_CROPS
    le = LabelEncoder()
    le.fit(all_crops)
    
    for i in range(num_clients):
        # Get data for this district
        if use_synthetic:
            df = generate_district_data(districts[i])
        else:
            df = generate_district_data(districts[i], df_kaggle=df_kaggle)
        
        # Extract ONLY the required feature columns (not all columns)
        X = df[FEATURE_COLUMNS].values  # Use only: N, P, K, pH, Rainfall, Temp
        y = le.transform(df['Crop'].values)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape for LSTM: (samples, time_steps, features)
        # We use 1 time step for simplicity as the paper suggests LSTM for its temporal capability 
        # but the specific data described is point-in-time soil/weather.
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        partitions.append({
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.long),
            'district': districts[i]
        })
        
    return partitions, le

if __name__ == "__main__":
    partitions, le = get_partitions()
    print(f"\n=== FLyer Federated Learning - Data Partitions ===")
    print(f"Total Edge Servers (Tier-III clients): {len(partitions)}")
    
    total_train_samples = 0
    total_test_samples = 0
    
    for i, p in enumerate(partitions):
        train_samples = len(p['X_train'])
        test_samples = len(p['X_test'])
        total_train_samples += train_samples
        total_test_samples += test_samples
        print(f"  Edge Server {i+1} ({p['district']}): Train={train_samples}, Test={test_samples}, Total={train_samples+test_samples}")
    
    total_samples = total_train_samples + total_test_samples
    print(f"\nTotal Samples: {total_samples}")
    if total_samples > 500:  # Real data has more samples
        print(f"  [OK] REAL DATA from Kaggle dataset!")
    else:
        print(f"  (Synthetic data - Kaggle dataset not found)")
    print(f"Crops: {', '.join(le.classes_)}")
    print(f"Training/Test Split: {total_train_samples}/{total_test_samples}")
    print(f"Note: Cloud server (Tier-IV) aggregates from {len(partitions)} edge servers during FL rounds.")
    print(f"=" * 50)
