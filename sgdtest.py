"""
Simple MLP Training Script for MNIST Classification
Architecture: Input -> 256-hidden (ReLU) -> Output (Softmax)
Loss: Cross-Entropy
Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Data loading with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        
        # Input layer (flattened 28x28 image)
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Hidden layer with ReLU activation
        self.relu = nn.ReLU()
        
        # Output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Flatten input (if not already)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model
model = MLP(input_size=784, hidden_size=64, num_classes=10).to(device)

# Loss function (Cross-Entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# Training loop
num_epochs = 5
print("\n" + "="*60)
print("Starting Training...")
print("="*60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_acc = evaluate(model, test_loader, device)
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

# Final evaluation
print("\n" + "="*60)
print("Final Evaluation")
print("="*60)
final_acc = evaluate(model, test_loader, device)
print(f"Final Test Accuracy: {final_acc:.2f}%")
print("="*60)

# Print model summary
print("\nModel Architecture:")
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Save model
torch.save(model.state_dict(), 'mnist_mlp_256.pth')
print("\nModel saved to 'mnist_mlp_256.pth'")