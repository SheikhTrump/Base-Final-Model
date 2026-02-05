import pickle
import matplotlib.pyplot as plt
import os
import sys

def visualize_results():
    # Load results
    results_path = "simulation_results.pkl"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run 'python simulate.py' first.")
        return

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    global_history = data["global_history"]
    client_accuracies = data["client_accuracies"]
    num_rounds = data["num_rounds"]
    
    # Extract Global Metrics
    rounds = [h["round"] for h in global_history]
    global_acc = [h["accuracy"] * 100 for h in global_history] # Convert to %
    
    # Setup Plot
    plt.figure(figsize=(12, 8))
    
    # Plot Global Accuracy
    plt.plot(rounds, global_acc, label=f"Global Server (Aggregated)", 
             color='black', linewidth=3, linestyle='--', marker='o')

    # Plot Each Client's Accuracy
    # Define some distinct colors for clients
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, cid in enumerate(sorted(client_accuracies.keys())):
        accs = client_accuracies[cid]
        # Client accuracies might be collected after each fit round.
        # Assuming 1:1 mapping with rounds if fraction_fit=1.0
        # If lengths differ, we align by round index
        client_rounds = range(1, len(accs) + 1)
        client_acc_percent = [a * 100 for a in accs]
        
        color = colors[i % len(colors)]
        # Use 'i + 1' to label clients sequentially (1 to 5) instead of the raw CID
        plt.plot(client_rounds, client_acc_percent, label=f"Edge Server {i+1}", 
                 color=color, linewidth=1.5, marker='x', alpha=0.7)

    # Formatting
    plt.title(f"Federated Learning Accuracy: Global Model vs. Local Clients\n({data['num_clients']} Clients, {num_rounds} Rounds)", fontsize=16)
    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    # Move legend inside the graph (best location automatically chosen)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save
    output_file = "accuracy_over_rounds.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved to: {os.path.abspath(output_file)}")
    
    # Show (optional, if supported)
    # plt.show()

if __name__ == "__main__":
    visualize_results()
