import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import CIFAR10Dataset
from model import CIFAR10Net
from tqdm import tqdm
from torchsummary import summary

def train_model(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f'Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Loading
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Model
    model = CIFAR10Net().to(device)
    
    # Print detailed model summary
    print("\nModel Summary:")
    summary(model, input_size=(3, 32, 32))
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')
    
    # Training
    train_model(model, train_loader, test_loader, epochs=50, device=device)

if __name__ == '__main__':
    main() 