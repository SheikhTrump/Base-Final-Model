from data_prep import get_partitions
import json
import os
import sys

# Force stdout flushing
sys.stdout.reconfigure(encoding='utf-8')

print("Starting verification...", flush=True)

try:
    # Load config to check num_clients
    try:
        with open('sim_config.json', 'r') as f:
            config = json.load(f)
        num_clients = config.get('simulation', {}).get('num_clients', 5)
        print(f"Config loaded. num_clients={num_clients}")
    except:
        num_clients = 5
        print("Config load failed. Using default num_clients=5")

    partitions, le = get_partitions(num_clients=num_clients)
    
    print("\nDatasample counts per client:")
    for i, p in enumerate(partitions):
        total = len(p['X_train']) + len(p['X_test'])
        dist = p.get('district', 'N/A')
        print(f"Client {i+1} [{dist}]: {total} samples")
        
except Exception as e:
    print(f"Error executing data_prep: {e}")
    import traceback
    traceback.print_exc()
