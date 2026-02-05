import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time
import flwr as fl
import flwr.simulation 
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from data_prep import get_partitions
from client import client_fn
from model import CropLSTM, get_parameters, set_parameters

import json

# Load Configuration
try:
    with open('sim_config.json', 'r') as f:
        CONFIG = json.load(f)
    print("  [OK] Loaded sim_config.json")
except FileNotFoundError:
    print("  [WARNING] sim_config.json not found. Using defaults.")
    # Default fallback structure matches new schema
    CONFIG = {
        "edge_servers": [
            {"name": "Default Edge 1", "processor": "Generic CPU"}
        ],
        "cloud_server": {"name": "Default Cloud", "ram": "16GB"},
        "network": {"uplink_mbps": 5, "downlink_mbps": 10},
        "simulation": {"num_rounds": 15, "num_clients": 4}
    }

# Network Latency Parameters
# Robustly get network settings, defaulting if keys missing
network_conf = CONFIG.get("network", {})
UPLINK_MBPS = network_conf.get("uplink_mbps", 5)
DOWNLINK_MBPS = network_conf.get("downlink_mbps", 10)

# Hardware Specifications (Simulated)
# The new config uses descriptive strings (e.g. "Intel Xeon") rather than explicit numeric limits.
# For the Flower Ray backend, we still need to strictly define resources.
# We will default to 1 CPU per client to ensure smooth execution on typical hardware.
CLIENT_RESOURCES = {
    "num_cpus": 1.0,
    "num_gpus": 0.0
}


def estimate_network_latency(data_size_bytes, direction='uplink'):
    """
    Estimate network transmission latency based on data size and network speed.
    Paper specifies: Uplink 5 Mbps, Downlink 10 Mbps
    Returns latency in milliseconds
    """
    if direction == 'uplink':
        speed_mbps = UPLINK_MBPS
    else:
        speed_mbps = DOWNLINK_MBPS
    
    # Convert bytes to megabits: (bytes * 8 bits/byte) / (1024*1024 bits/MB)
    data_mb = (data_size_bytes * 8) / (1024 * 1024)
    
    # Latency = data_mb / speed_mbps * 1000 ms/second
    latency_ms = (data_mb / speed_mbps) * 1000
    return latency_ms

class CustomSimStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_network_latency = 0
        self.round_history = []
        self.client_accuracies = {} # cid -> list of accuracies
        # Store initial parameters for size calculation
        # Will be updated by run_simulation to match actual class count
        self.model = CropLSTM(num_classes=2) 
        self.param_size_bytes = sum([p.nbytes for p in get_parameters(self.model)])
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure the next round of training.
        """
        # Call safe super method
        config = super().configure_fit(server_round, parameters, client_manager)
        if not config:
            return []

        # Get local_epochs from config (use first edge server as reference or default to 5)
        edge_servers = CONFIG.get("edge_servers", [])
        if edge_servers:
            local_epochs = edge_servers[0].get("local_epochs", 5)
        else:
            local_epochs = 5
            
        # Add to configuration
        for i in range(len(config)):
            original_tuple = config[i] # (ClientProxy, FitIns)
            client_proxy = original_tuple[0]
            fit_ins = original_tuple[1]
            
            # Create new config dict merging existing and new
            new_config = fit_ins.config.copy()
            new_config["local_epochs"] = local_epochs
            
            # Create new FitIns
            new_fit_ins = fl.common.FitIns(fit_ins.parameters, new_config)
            
            # Replace in list
            config[i] = (client_proxy, new_fit_ins)
            
        # Calculate Downlink Latency (Server -> Client)
        # This happens for EACH client selected
        num_clients = len(config)
        
        # Estimate size of parameters being sent
        # simpler estimation using our local model
        uplink_latency = estimate_network_latency(self.param_size_bytes, 'uplink')
        downlink_latency = estimate_network_latency(self.param_size_bytes, 'downlink')
        
        latency_this_round = downlink_latency * num_clients
        self.total_network_latency += latency_this_round
        
        # We can also store this to print later
        self.current_round_downlink = latency_this_round
        
        return config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using weighted average and track custom metrics.
        """
        if not results:
            return None, {}
            
        # 1. Standard FedAvg Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is None:
            return None, {}

        # 2. Custom Metric Aggregation
        # Extract metrics from FitRes
        # Structure of results: [(ClientProxy, FitRes), ...]
        
        total_examples = sum([res.num_examples for _, res in results])
        
        # Weighted averages for accuracy/custom metrics
        weighted_acc = sum([res.metrics["accuracy"] * res.num_examples for _, res in results]) / total_examples
        weighted_f1 = sum([res.metrics["f1"] * res.num_examples for _, res in results]) / total_examples
        weighted_enc = sum([res.metrics["enc_time"] * res.num_examples for _, res in results]) / total_examples
        weighted_dec = sum([res.metrics["dec_time"] * res.num_examples for _, res in results]) / total_examples
        weighted_data_enc = sum([res.metrics["data_enc_time"] * res.num_examples for _, res in results]) / total_examples

        # Store individual client accuracies across rounds
        for proxy, res in results:
            if "accuracy" in res.metrics:
                cid = proxy.cid
                acc = res.metrics["accuracy"]
                if cid not in self.client_accuracies:
                    self.client_accuracies[cid] = []
                self.client_accuracies[cid].append(acc)

        # 3. Calculate Uplink Latency (Client -> Server)
        # For each client result received
        uplink_latency_one = estimate_network_latency(self.param_size_bytes, 'uplink')
        uplink_latency_total = uplink_latency_one * len(results)
        
        self.total_network_latency += uplink_latency_total
        
        round_network_latency = self.current_round_downlink + uplink_latency_total
        
        print(f"Round {server_round} Aggregation:")
        print(f"  Accuracy: {weighted_acc*100:.2f}% | F1: {weighted_f1:.4f}")
        print(f"  Encryption (Grad/Data): {weighted_enc:.4f}/{weighted_data_enc:.4f} ms")
        print(f"  Network Latency (Round): {round_network_latency:.2f} ms")
        
        # Store for history
        self.round_history.append({
            "round": server_round,
            "accuracy": weighted_acc,
            "f1": weighted_f1,
            "enc_time": weighted_enc,
            "dec_time": weighted_dec,
            "data_enc_time": weighted_data_enc,
            "network_latency": round_network_latency
        })
        
        # Return aggregated parameters and our custom metrics
        return aggregated_parameters, {
            "accuracy": weighted_acc,
            "f1": weighted_f1,
            "enc_time": weighted_enc,
            "dec_time": weighted_dec,
            "data_enc_time": weighted_data_enc,
            "round_network_latency": round_network_latency
        }

def run_simulation():
    """
    Orchestrates the Flower simulation using the Virtual Client Engine.
    """
    print("Starting FLyer Federated Learning Simulation (Flower Virtual Engine)...")
    print("=" * 70)
    print("HARDWARE CONFIGURATION (Simulated Architecture)")
    print("-" * 40)
    
    # Cloud Server Info
    cloud = CONFIG.get("cloud_server", {})
    print("1. Tier-IV Cloud Server (Global Model):")
    print(f"   - Name: {cloud.get('name', 'Cloud Server')}")
    print(f"   - Spec: {cloud.get('processor', 'N/A')} | RAM: {cloud.get('ram', 'N/A')}")

    # Edge Servers Info (Iterate list)
    edge_servers = CONFIG.get("edge_servers", [])
    print(f"\n2. Tier-III Edge Servers ({len(edge_servers)} Configured):")
    # Just print the first one as an example to avoid clutter, or summary
    if edge_servers:
        first_edge = edge_servers[0]
        print(f"   - Example Node: {first_edge.get('name', 'Edge Node')}")
        print(f"   - Processor: {first_edge.get('processor', 'N/A')}")
        print(f"   - RAM: {first_edge.get('ram', 'N/A')} | Storage: {first_edge.get('storage', 'N/A')}")
        print(f"   - Resources allocated (Sim): {CLIENT_RESOURCES}")
    else:
        print("   - [WARNING] No edge servers defined in config.")

    print("\n3. Network Simulation:")
    print(f"   - Uplink: {UPLINK_MBPS} Mbps")
    print(f"   - Downlink: {DOWNLINK_MBPS} Mbps")
    print("=" * 70)
    
    # 1. Load Data
    # Use simulation settings for control
    sim_settings = CONFIG.get("simulation", {})
    num_clients = sim_settings.get("num_clients", 4)
    num_rounds = sim_settings.get("num_rounds", 15)
    partitions, le = get_partitions(num_clients=num_clients)
    
    # Determine number of classes dynamically (Multi-class support)
    num_classes = len(le.classes_)
    print(f"\n[INFO] Model Configuration: Output Classes = {num_classes} ({', '.join(le.classes_)})")

    # Track sample counts for reporting
    client_sample_counts = {}
    for i, p in enumerate(partitions):
        count = len(p['X_train']) + len(p['X_test'])
        client_sample_counts[f"Edge Server {i+1}"] = count
        
    # Note: Device determination is handled inside FlowerClient
    
    # 2. Define Client Function for Simulation
    # fl.simulation.start_simulation expects a function that takes `cid`
    def client_fn_simulation(cid: str) -> fl.client.Client:
        # We pass "cpu" merely as a placeholder if client_fn requires it, 
        # but client.py's client_fn doesn't actually use the device arg for logic.
        # NOW: We must also pass num_classes to ensure local models are built correctly
        return client_fn(cid, partitions, "cpu", num_classes=num_classes)

    # 3. Initialize Global Model (Server Side)
    global_model = CropLSTM(num_classes=num_classes)
    global_parameters = ndarrays_to_parameters(get_parameters(global_model))

    # 4. Initialize Strategy
    strategy = CustomSimStrategy(
        fraction_fit=1.0,  # Sample 100% of clients
        fraction_evaluate=0.0, # No distributed evaluation needed for now
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=global_parameters,
    )
    
    # Store initial parameters for size calculation (Needs updated model size)
    strategy.model = CropLSTM(num_classes=num_classes)
    strategy.param_size_bytes = sum([p.nbytes for p in get_parameters(strategy.model)])
    
    start_time = time.time()
    
    print(f"\nStarting Simulation for {num_rounds} rounds with {num_clients} clients...")
    
    # --- START SIMULATION ---
    # We use start_simulation which handles the loop and client resource management
    fl.simulation.start_simulation(
        client_fn=client_fn_simulation,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
    )
    
    # --- END OF SIMULATION ---
    
    total_duration = time.time() - start_time
    
    # 5. Final Reporting & Visualization
    print("\\n" + "="*70)
    print("FINAL SIMULATION RESULTS")
    print("="*70)
    
    # Extract metrics from strategy history
    history = strategy.round_history
    if not history:
        print("No history recorded. Simulation might have failed or ran 0 rounds.")
        return

    final_acc = history[-1]["accuracy"]
    weighted_f1 = history[-1]["f1"]
    
    print(f"\\n>>> ACCURACY METRICS <<<")
    print(f"Final Global Accuracy: {final_acc*100:.2f}%")
    print(f"Final Global F1: {weighted_f1:.4f}")

    print(f"\n>>> EDGE SERVER (CLIENT) ACCURACY <<<")
    # Print per-client final accuracy
    if hasattr(strategy, 'client_accuracies'):
        sorted_cids = sorted(strategy.client_accuracies.keys())
        for i, cid in enumerate(sorted_cids):
            accs = strategy.client_accuracies[cid]
            if accs:
                final_client_acc = accs[-1]
                print(f"  Edge Server {i+1} Final Accuracy: {final_client_acc*100:.2f}%")
    
    # Calculate Averages
    avg_enc = sum(h["enc_time"] for h in history) / len(history)
    avg_dec = sum(h["dec_time"] for h in history) / len(history)
    avg_data_enc = sum(h["data_enc_time"] for h in history) / len(history)
    avg_network = sum(h["network_latency"] for h in history) / len(history)
    
    print(f"\\n>>> LATENCY METRICS <<<")
    print(f"Average Encryption Latency Per Round (Per Client):")
    print(f"  - Gradient Encryption: {avg_enc:.4f} ms")
    print(f"  - Gradient Decryption: {avg_dec:.4f} ms")
    print(f"  - Data Encryption (AES-256): {avg_data_enc:.4f} ms")
    
    print(f"\\nNetwork Transmission Latency (Per Round):")
    print(f"  - Average: {avg_network:.4f} ms")
    print(f"  - Total (All Rounds): {strategy.total_network_latency:.2f} ms")
    
    # Save Results for Visualization
    print(f"\\nSaving simulation results to 'simulation_results.pkl'...")
    results_data = {
        "global_history": strategy.round_history,
        "client_accuracies": strategy.client_accuracies,
        "num_rounds": num_rounds,
        "num_clients": num_clients
    }
    
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(results_data, f)
    print("Results saved successfully.")
    
    print(f"\\n>>> SYSTEM PERFORMANCE <<<")
    print(f"Total Model Simulation Time: {total_duration:.2f} seconds")
    
    # Energy Estimation
    total_energy = total_duration * 5.0 # 5W
    print(f"Estimated Total Energy (5W edge device): {total_energy:.2f} Joules")

    # ========================================================================
    # SAVE RESULTS FOR VISUALIZATION
    # ========================================================================
    print(f"\\nSaving detailed results to 'sim_results.json'...")
    
    # Prepare data structure
    results_data = {
        "rounds": [h["round"] for h in history],
        "accuracy": [h["accuracy"] * 100 for h in history], # Convert to %
        "f1": [h["f1"] for h in history],
        "enc_time": [h["enc_time"] for h in history],
        "dec_time": [h["dec_time"] for h in history],
        "data_enc_time": [h["data_enc_time"] for h in history],
        "network_latency": [h["network_latency"] for h in history],
        "local_accuracies": {},
        "client_sample_counts": client_sample_counts,
        "total_energy": total_energy,
        "total_time": total_duration,
        "metrics": {
            "avg_enc": avg_enc,
            "avg_data_enc": avg_data_enc,
            "avg_network": avg_network
        }
    }
    
    # Add local accuracies
    if hasattr(strategy, 'client_accuracies'):
        sorted_cids = sorted(strategy.client_accuracies.keys())
        for i, cid in enumerate(sorted_cids):
            accs = strategy.client_accuracies[cid]
            if accs:
                results_data["local_accuracies"][f"Edge Server {i+1}"] = accs[-1] * 100
                
    # Save to file
    import json
    with open('sim_results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"[OK] Results saved. Run 'python visualize_results.py' to generate graphs.")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
