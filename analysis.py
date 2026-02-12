import torch
from PopMLP import PopMLP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Load MNIST dataset (same as in para.py)
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convert labels to one-hot encoding
y = y.astype(int)
y_onehot = np.zeros((y.shape[0], 10))
y_onehot[np.arange(y.shape[0]), y] = 1

# Split into train and test sets (using same split as para.py)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=1/7, random_state=42)

# Convert to torch tensors
x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Check for GPU availability and move tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x_test = x_test.to(device)
y_test = y_test.to(device)

# Define network parameters (same as in para.py)
shapes = [784, 1000, 10]
activation = torch.relu
output_activation = lambda x: x
precision = 'i4'
bias_std = 0.
mutation_std = torch.pi/20
scale_std = 0.

# Load the saved population
print("Loading saved population...")
population_size = 500
pop_mlp = PopMLP(population_size, shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std)

# Load state dict from file
state_dict = torch.load('final_pop.npy', map_location=device)
pop_mlp.load_state_dict(state_dict)

print("Population loaded successfully.")

def accuracy(logits, targets):

    logits_flat = logits.reshape(-1, logits.size(-1))  # (networks*batch_size, classes)
    targets_flat = targets.reshape(-1, targets.size(-1))  # (networks*batch_size, classes)

    acc_per_sample = (logits_flat.argmax(dim=1) == targets_flat.argmax(dim=1)).float()

    acc_per_sample = acc_per_sample.reshape(logits.size(0), logits.size(1))

    return acc_per_sample.mean(dim=1)

# Run forward pass on test dataset
print("Running forward pass...")
accuracies = torch.zeros(population_size).to(device=device)
with torch.no_grad():
    for i in range(10):
        acc = pop_mlp.evaluate(x_test[i*1000:(i + 1)*1000], y_test[i*1000:(i + 1)*1000], accuracy)
        accuracies += acc/10

# Report metrics
print(f"Number of individuals in population: {population_size}")
print(f"Test samples used: {len(x_test)}")
print(f"Mean Accuracy: {accuracies.mean().item():.4f}")
print(f"Best Accuracy: {accuracies.max().item():.4f}")
print(f"Worst Accuracy: {accuracies.min().item():.4f}")
print(f"Std Dev Accuracy: {accuracies.std().item():.4f}")

# Create histogram of accuracies
plt.figure(figsize=(10, 6))
plt.hist(accuracies.cpu().numpy(), bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Individual Accuracies Across Population')
plt.xlabel('Accuracy')
plt.ylabel('Number of Individuals')
plt.grid(True, alpha=0.3)

# Add statistics to the plot
plt.axvline(accuracies.mean().item(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {accuracies.mean().item():.4f}')
plt.axvline(accuracies.max().item(), color='green', linestyle=':', linewidth=2, 
           label=f'Best: {accuracies.max().item():.4f}')
plt.axvline(accuracies.min().item(), color='orange', linestyle=':', linewidth=2, 
           label=f'Worst: {accuracies.min().item():.4f}')

plt.legend()
plt.tight_layout()
plt.savefig('accuracy_histogram.png')
plt.close()

print("Accuracy histogram saved as 'accuracy_histogram.png'")
'''
# Show some sample predictions
print("\nSample predictions:")
for i in range(5):
    individual_idx = torch.randint(0, population_size, (1,)).item()
    sample_idx = torch.randint(0, len(x_test), (1,)).item()
    predicted_class = logits[individual_idx, sample_idx].argmax().item()
    true_class = y_test[sample_idx].argmax().item()
    accuracy_for_individual = accuracies[individual_idx].item()
    
    print(f"Individual {individual_idx}: Sample {sample_idx} - Predicted: {predicted_class}, "
          f"True: {true_class}, Accuracy: {accuracy_for_individual:.4f}")
    '''