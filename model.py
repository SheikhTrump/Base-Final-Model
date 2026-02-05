import torch
import torch.nn as nn
from collections import OrderedDict

class CropLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, num_classes=2):
        super(CropLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Table II Architecture Specifications (from FLyer Paper):
        # LSTM unit: 64
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # First Dense layer unit: 128, Dropout: 0.2
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second Dense layer unit: 64, Dropout: 0.4
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        
        # Output layer: 16 units for ALL crops in dataset (varies by district)
        # Paper shows each district uses subset of crops (7-12 crops per district)
        # Global model uses all 16 crops for FedAvg compatibility
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        # Dense Layers
        out = torch.relu(self.fc1(out))
        out = self.dropout1(out)
        
        out = torch.relu(self.fc2(out))
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, optimizer, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device, crop_names=None):
    """Validate the network on the entire test set.
    
    Parameters:
    -----------
    net : CropLSTM
        The model to test
    testloader : DataLoader
        Test data loader
    device : torch.device
        Device to run on (CPU or GPU)
    crop_names : list, optional
        List of crop names for this district (if None, uses all 16 global crops)
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    net.eval()
    
    # Calculate additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    # Default to all 16 crops if not specified
    if crop_names is None:
        crop_names = ["Sugarcane", "Jowar", "Cotton", "Rice", "Wheat", "Groundnut", 
                     "Maize", "Tur", "Urad", "Moong", "Gram", "Masoor", 
                     "Soybean", "Ginger", "Turmeric", "Grapes"]
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Print predictions with crop names (show first 20)
    print(f"\n  Predictions Summary (District-specific crops):")
    print(f"  {'-' * 90}")
    for i, (actual, pred) in enumerate(zip(all_labels[:20], all_preds[:20])):  # Print first 20
        if int(actual) < len(crop_names):
            actual_name = crop_names[int(actual)]
        else:
            actual_name = f"Unknown({actual})"
        if int(pred) < len(crop_names):
            pred_name = crop_names[int(pred)]
        else:
            pred_name = f"Unknown({pred})"
        match = "✓" if actual == pred else "✗"
        print(f"    [{i+1:2d}] Actual: {actual_name:15s} | Predicted: {pred_name:15s} | {match}")
    if len(all_labels) > 20:
        print(f"    ... and {len(all_labels) - 20} more samples")
    print(f"  {'-' * 90}\n")
    
    return loss, accuracy, f1, precision, recall
