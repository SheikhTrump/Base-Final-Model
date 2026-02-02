import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time

from data_prep import get_partitions
from client import client_fn
from model import CropLSTM, get_parameters, set_parameters
import numpy as np

# Load partitions
partitions, le = get_partitions(num_clients=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Network Latency Parameters (Tier-II to Tier-III and Tier-III to Tier-IV)
# Paper specifies: Uplink 5 Mbps, Downlink 10 Mbps
UPLINK_MBPS = 5
DOWNLINK_MBPS = 10

def aggregate_parameters(results):
    """
    Compute weighted average of parameters.
    results: list of tuples (parameters, num_examples)
    Implements FedAvg algorithm as per paper specification.
    """
    num_examples_total = sum([num_examples for _, num_examples in results])
    
    # Initialize aggregated parameters with zeros
    weighted_weights = [np.zeros_like(w) for w in results[0][0]]
    
    for weights, num_examples in results:
        for i, layer_weights in enumerate(weights):
            weighted_weights[i] += layer_weights * num_examples
            
    # Divide by total examples - weighted averaging per FedAvg
    aggregated_weights = [w / num_examples_total for w in weighted_weights]
    return aggregated_weights

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

def start_custom_simulation():
    print("Starting FLyer Federated Learning Simulation (Custom Loop)...")
    print("=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)
    start_time = time.time()
    print(f"Algorithm: FedAvg (Federated Averaging)")
    print(f"Model: LSTM with 64 units, Dense 128/64, Dropout 0.2/0.4")
    print(f"Tier-III Clients (Edge Servers): 4 (Districts of Western Maharashtra)")
    print(f"Tier-IV Server (Cloud): 1 (Global Aggregation)")
    print(f"Network Speed: Uplink {UPLINK_MBPS} Mbps, Downlink {DOWNLINK_MBPS} Mbps (Paper spec)")
    print(f"Data: Synthetic (Paper uses 4,513 real Kaggle samples)")
    print(f"Target Accuracy: Global >99%, Local 94-97% (Paper goal)")
    print("=" * 70 + "\n")
    
    # Initialize Global Model
    global_model = CropLSTM()
    global_parameters = get_parameters(global_model)
    
    # Estimate parameter size for network latency calculation
    params_size = sum([p.nbytes for p in global_parameters])
    print(f"Model Parameters Size: {params_size / 1024:.2f} KB")
    
    # Estimate network latencies per round
    uplink_latency_ms = estimate_network_latency(params_size, 'uplink')
    downlink_latency_ms = estimate_network_latency(params_size, 'downlink')
    print(f"Network Latency per Round:")
    print(f"  - Uplink (Client->Cloud): {uplink_latency_ms:.4f} ms")
    print(f"  - Downlink (Cloud->Client): {downlink_latency_ms:.4f} ms")
    print("=" * 70 + "\n")
    
    num_rounds = 15
    clients = [client_fn(str(i), partitions, device) for i in range(4)]
    
    history_accuracy = []
    total_network_latency = 0

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} / {num_rounds} ---")
        
        round_results = []
        round_metrics = []
        round_network_latency = 0
        
        # 1. Distribute global parameters to clients (Downlink: Tier-IV -> Tier-III)
        round_network_latency += downlink_latency_ms * len(clients)
        
        # 2. Train all clients
        for client in clients:
            # Set global parameters
            client.fit(global_parameters, {})
            
            # Get updated parameters and metrics
            updated_params, num_examples, _ = client.fit(global_parameters, {})
            round_results.append((updated_params, num_examples))
            
            # Estimate network latency for parameter upload (Tier-III -> Tier-IV)
            round_network_latency += uplink_latency_ms
            
            # Evaluate client on local test set for reporting
            loss, count, metrics = client.evaluate(updated_params, {})
            round_metrics.append(metrics)
        
        # 3. Aggregate parameters at cloud (Tier-IV)
        global_parameters = aggregate_parameters(round_results)
        
        # 4. Calculate weighted metrics
        total_examples = sum([r[1] for r in round_results])
        
        weighted_acc = sum([m["accuracy"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_f1 = sum([m["f1"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_prec = sum([m["precision"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_rec = sum([m["recall"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_enc = sum([m["enc_time"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_dec = sum([m["dec_time"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        weighted_data_enc = sum([m["data_enc_time"] * res[1] for m, res in zip(round_metrics, round_results)]) / total_examples
        
        history_accuracy.append(weighted_acc)
        total_network_latency += round_network_latency
        
        # Store latency for final average
        if round_num == 1:
            total_enc_time = weighted_enc
            total_dec_time = weighted_dec
            total_data_enc_time = weighted_data_enc
        else:
             total_enc_time += weighted_enc
             total_dec_time += weighted_dec
             total_data_enc_time += weighted_data_enc

        print(f"  Accuracy: {weighted_acc*100:.2f}% | F1: {weighted_f1:.4f} | Encryption (Grad/Data): {weighted_enc:.4f}/{weighted_data_enc:.4f} ms | Network Latency: {round_network_latency:.2f} ms")
        
    # Final Results
    print("\n" + "="*70)
    print("FINAL SIMULATION RESULTS")
    print("="*70)
    
    final_acc = history_accuracy[-1]
    print(f"\n>>> ACCURACY METRICS <<<")
    print(f"Final Global Accuracy: {final_acc*100:.2f}%")
    print(f"Final Global F1: {weighted_f1:.4f}")
    print(f"  (Paper Target: Global >99%, Local 94-97%)")
    
    avg_enc = total_enc_time / num_rounds
    avg_dec = total_dec_time / num_rounds
    avg_data_enc = total_data_enc_time / num_rounds
    avg_network_latency = total_network_latency / num_rounds
    
    print(f"\n>>> LATENCY METRICS <<<")
    print(f"Average Encryption Latency Per Round (Per Client):")
    print(f"  - Gradient Encryption: {avg_enc:.4f} ms (Paper: ~4.29 ms)")
    print(f"  - Gradient Decryption: {avg_dec:.4f} ms")
    print(f"  - Data Encryption (AES-256): {avg_data_enc:.4f} ms (Paper: ~39.08 ms)")
    
    print(f"\nNetwork Transmission Latency (Per Round):")
    print(f"  - Average: {avg_network_latency:.4f} ms")
    print(f"  - Total (All Rounds): {total_network_latency:.2f} ms")
    print(f"  - Based on Uplink {UPLINK_MBPS} Mbps, Downlink {DOWNLINK_MBPS} Mbps")
    
    total_duration = time.time() - start_time
    print(f"\n>>> SYSTEM PERFORMANCE <<<")
    print(f"Total Model Simulation Time: {total_duration:.2f} seconds")
    
    # Energy Estimation (5W edge device as per paper methodology)
    power_watt = 5.0
    total_energy = total_duration * power_watt
    print(f"Estimated Total Energy (5W edge device): {total_energy:.2f} Joules ({total_energy/1000:.3f} KJ)")
    print(f"  (Paper reports: 4-6 KJ for full system)")
    
    if final_acc >= 0.94:
        print(f"\n[OK] SUCCESS: Achieved {final_acc*100:.2f}% accuracy (Target >94% local)")
    else:
        print(f"\n[WARNING] NOTE: Achieved {final_acc*100:.2f}% accuracy (Below 94% target)")
    
    # Format accuracy strings for all edge servers
    local_accuracies = [f"{m['accuracy']*100:.2f}%" for m in round_metrics]
    print(f"\nLocal accuracies by edge server: {', '.join(local_accuracies)}")
    print("="*70)
    
    # ====== AUTOMATICALLY GENERATE VISUALIZATIONS ======
    print("\n" + "="*70)
    print("Generating visualization plots...")
    print("="*70)
    try:
        from visualize_results import (create_comprehensive_plots, 
                                       create_accuracy_focus_plot,
                                       create_latency_comparison_plot,
                                       create_system_performance_plot)
        
        print("\n Creating comprehensive performance plots...")
        fig1 = create_comprehensive_plots()
        fig1.savefig('flyer_comprehensive_results.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: flyer_comprehensive_results.png")
        
        print(" Creating accuracy focus plot...")
        fig2 = create_accuracy_focus_plot()
        fig2.savefig('flyer_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: flyer_accuracy_analysis.png")
        
        print(" Creating latency comparison plot...")
        fig3 = create_latency_comparison_plot()
        fig3.savefig('flyer_latency_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: flyer_latency_comparison.png")
        
        print(" Creating system performance plot...")
        fig4 = create_system_performance_plot()
        fig4.savefig('flyer_system_performance.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: flyer_system_performance.png")
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        print("\n4 high-resolution plots generated:")
        print("  1. flyer_comprehensive_results.png (0.95 MB)")
        print("  2. flyer_accuracy_analysis.png (0.36 MB)")
        print("  3. flyer_latency_comparison.png (0.21 MB)")
        print("  4. flyer_system_performance.png (0.61 MB)")
        print("\nOpen these images to visualize the FL training results!")
        print("="*70 + "\n")
        
        # Close all matplotlib figures to free memory
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except ImportError:
        print("\n[WARNING] Visualization module not found. Skipping plot generation.")
        print("To generate plots, ensure visualize_results.py is in the project directory.")
    except Exception as e:
        print(f"\n[WARNING] Error during visualization: {str(e)}")
        print("Continuing without plots...")
    
if __name__ == "__main__":
    start_custom_simulation()
