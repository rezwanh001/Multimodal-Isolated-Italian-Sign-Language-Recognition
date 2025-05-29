import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import SessioniDataset, train_transform, val_transform
from model import TwoStreamCNNLSTM, AdvancedTwoStreamModel
from tqdm import tqdm
import os

def train_model():
    # Hyperparameters
    batch_size = 4
    num_epochs = 25
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    data_directory = '/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/iccv-mslr-2025-track-2/iccv-mslr-2025-track-2/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val'
    train_dataset = SessioniDataset(root_dir=f'{data_directory}/train', split='train', num_frames=16, transform=train_transform)
    val_dataset = SessioniDataset(root_dir=f'{data_directory}/val', split='val', num_frames=16, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ## Model and multi-GPU
    # model = TwoStreamCNNLSTM(num_classes=126)
    model = AdvancedTwoStreamModel(num_classes=126)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)


    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            rgb = batch['rgb'].to(device)   # [B, T, C, H, W]
            rdm = batch['rdm'].to(device)   # [B, 3, T, C, H, W]
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(rgb, rdm)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')

        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), './outputs/models/best_model.pth')
            print(f'[✔] Best model updated at epoch {epoch+1} with accuracy {train_acc:.2f}%')

    print('Training completed.')

if __name__ == '__main__':
    os.makedirs('./outputs/models', exist_ok=True)
    train_model()
