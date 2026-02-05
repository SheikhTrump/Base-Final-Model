from data_prep import get_partitions
import torch
import pandas as pd
import sys

# Force stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

def show_client_data(client_idx=0):
    print(f"\nFETCHING DATA FOR CLIENT {client_idx + 1}...", flush=True)
    
    # Get partitions
    # Note: This runs the whole pipeline again, so it might take a second
    partitions, le_global = get_partitions(num_clients=5, run_eda=False)
    
    p = partitions[client_idx]
    
    # Combine Train and Test to see "ALL" 1000 samples
    X_train = p['X_train']
    y_train = p['y_train']
    X_test = p['X_test']
    y_test = p['y_test']
    
    # Concatenate
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    # Reshape back from (Batch, 1, 6) to (Batch, 6) for display
    X_flat = X_all.reshape(X_all.shape[0], X_all.shape[1] * X_all.shape[2])
    
    # Convert to Numpy
    X_np = X_flat.numpy()
    y_np = y_all.numpy()
    
    # Decode labels
    y_decoded = le_global.inverse_transform(y_np)
    
    # Create DataFrame for nice printing
    # Features are 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature'
    feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temp']
    
    df = pd.DataFrame(X_np, columns=feature_cols)
    df['Crop_Label'] = y_decoded
    
    print("\n" + "="*80)
    print(f"CLIENT {client_idx + 1} COMPLETE DATASET ({p['district']})")
    print(f"Total Samples: {len(df)}")
    print("="*80)
    
    # Set pandas display options to show ALL rows
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string())
    print("\n" + "="*80)
    print(f"END OF DATA - Total {len(df)} rows displayed")
    print("="*80)

if __name__ == "__main__":
    show_client_data(0) # Show Client 1
