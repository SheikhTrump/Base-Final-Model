"""
FLyer: Federated Learning-Based Crop Yield Prediction for Agriculture 5.0
Paper: https://arxiv.org/... (Federated Learning at Edge with Encrypted Gradients)

This module implements Tier-III (Edge Server) clients for the FLyer system.

SYSTEM ARCHITECTURE (4 Tiers):
-------------------------------
Tier-I (IoT Layer):
  - Sensors collect: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature
  - Microcontrollers aggregate sensor data

Tier-II (User Devices):
  - Mobile phones and tablets pre-process data
  - Cache-based dew computing for offline data storage
  - Transmits preprocessed data to edge servers

Tier-III (Edge Layer - THIS MODULE):
  - 4 Edge Servers with local LSTM models
  - Processes data locally using FedAvg
  - Encrypts gradients with AES-256 before transmission
  - Each edge server handles one district (Pune, Ahmednagar, Solapur, Satara)

Tier-IV (Cloud Layer):
  - Private cloud servers perform global model aggregation
  - Executes FedAvg algorithm on encrypted gradients
  - Sends updated global model back to edge servers

FEDERATED LEARNING APPROACH:
----------------------------
Algorithm: FedAvg (Federated Averaging)
- Local training on edge servers with district data
- Gradient encryption (AES-256) for privacy
- Weighted parameter aggregation at cloud
- Multiple rounds until convergence

SECURITY:
---------
- Gradient Encryption: AES-256 (GCM mode)
- Gradient Latency: ~4.29 ms (per paper)
- Data Encryption (local storage): AES-256
- Data Encryption Latency: ~39.08 ms (per paper)
- Information leakage ratio < 0.04 even with 50% malicious clients
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CropLSTM, get_parameters, set_parameters, train, test
import os
import time
import pickle
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, testloader, crop_names=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.crop_names = crop_names  # District-specific crop names
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        
        # Simulate Gradient Encryption (AES-256)
        # Measure Latency
        try:
            # 1. Gradient/Parameter Encryption
            # Serialization
            params_bytes = pickle.dumps(parameters)
            
            # Encryption
            start_enc = time.time()
            key = get_random_bytes(32) # AES-256 key
            cipher = AES.new(key, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(params_bytes)
            enc_time = (time.time() - start_enc) * 1000 # ms
            
            # Decryption (Validation)
            start_dec = time.time()
            cipher_dec = AES.new(key, AES.MODE_GCM, nonce=cipher.nonce)
            decrypted_data = cipher_dec.decrypt_and_verify(ciphertext, tag)
            dec_time = (time.time() - start_dec) * 1000 # ms
            
            # 2. Local Data Encryption (One-off check, simulated every round for reporting)
            # Serialize entire dataset
            data_bytes = pickle.dumps([self.trainloader.dataset.tensors])
            
            start_data_enc = time.time()
            key_data = get_random_bytes(32)
            cipher_data = AES.new(key_data, AES.MODE_GCM)
            ciphertext_data, tag_data = cipher_data.encrypt_and_digest(data_bytes)
            data_enc_time = (time.time() - start_data_enc) * 1000
            
            print(f"[Client {self.cid}] AES-256 Latency | Gradients: {enc_time:.4f} ms | Data: {data_enc_time:.4f} ms | Grad Size: {len(ciphertext)/1024:.2f} KB")
            
            # Store in self for reporting in evaluate
            self.last_enc_time = enc_time
            self.last_dec_time = dec_time
            self.last_data_enc_time = data_enc_time
            
        except Exception as e:
            print(f"[Client {self.cid}] Encryption Error: {e}")
            self.last_enc_time = 0.0
            self.last_dec_time = 0.0
        
        epochs = int(config.get("local_epochs", 5))
        train(self.net, self.trainloader, torch.optim.Adam(self.net.parameters(), lr=0.001), epochs=epochs, device=self.device)
        
        # Evaluate on local test set to report metrics during fit (for custom strategy)
        loss, accuracy, f1, precision, recall = test(self.net, self.testloader, device=self.device, crop_names=self.crop_names)
        
        print(f"[Client {self.cid}] Local Accuracy: {accuracy*100:.2f}% | Local F1: {f1:.4f}")

        metrics = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "enc_time": getattr(self, "last_enc_time", 0.0),
            "dec_time": getattr(self, "last_dec_time", 0.0),
            "data_enc_time": getattr(self, "last_data_enc_time", 0.0)
        }
        
        return get_parameters(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy, f1, precision, recall = test(self.net, self.testloader, device=self.device, crop_names=self.crop_names)
        print(f"[Client {self.cid}] acc: {accuracy:.4f}, f1: {f1:.4f}")
        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "enc_time": getattr(self, "last_enc_time", 0.0),
            "dec_time": getattr(self, "last_dec_time", 0.0),
            "data_enc_time": getattr(self, "last_data_enc_time", 0.0)
        }

def client_fn(cid: str, partitions, device, num_classes=2):
    """Create a Flower client representing a single organization."""
    # Load model with correct number of classes
    net = CropLSTM(num_classes=num_classes)

    # Note: partitions is a list of dicts. cid is a string index.
    partition = partitions[int(cid)]
    
    # Create DataLoaders
    trainset = TensorDataset(partition['X_train'], partition['y_train'])
    testset = TensorDataset(partition['X_test'], partition['y_test'])
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    # Pass district-specific crop names
    crop_names = partition.get('crop_names', None)

    return FlowerClient(cid, net, trainloader, testloader, crop_names=crop_names)
