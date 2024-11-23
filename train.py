from dataset import TwoStreamDataset
from model import TwoStreamSNN
from torch.utils.data import DataLoader
import torch
from torch import nn
import os  
import time  
import matplotlib.pyplot as plt  
import torch.optim as optim  

def save_model(model, epoch, save_dir):  
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))  

def plot_and_save_loss(train_losses, val_losses, save_path):  
    plt.figure(figsize=(8, 6))  
    plt.plot(train_losses, label='Training Loss')  
    plt.plot(val_losses, label='Validation Loss')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.title('Training and Validation Loss')  
    plt.legend()  
    plt.savefig(save_path)  
    plt.close()  

def train_model(model, train_dataloader, val_dataloader, num_epochs, save_dir):  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    model.to(device)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    criterion = nn.CrossEntropyLoss()  

    train_losses = []  
    val_losses = []  

    for epoch in range(num_epochs):  
        start_time = time.time()  

        # 训练模式  
        model.train()  
        train_loss = 0.0  
        for batch in train_dataloader:  
            rgb_data, flow_data, labels = batch['rgb'].to(device), batch['flow'].to(device), batch['label'].to(device)  
            optimizer.zero_grad()  
            outputs = model(rgb_data, flow_data)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            train_loss += loss.item()  

        train_loss /= len(train_dataloader)  
        train_losses.append(train_loss)  

        # 验证模式  
        model.eval()  
        val_loss = 0.0  
        with torch.no_grad():  
            for batch in val_dataloader:  
                rgb_data, flow_data, labels = batch['rgb'].to(device), batch['flow'].to(device), batch['label'].to(device)  
                outputs = model(rgb_data, flow_data)  
                loss = criterion(outputs, labels)  
                val_loss += loss.item()  

        val_loss /= len(val_dataloader)  
        val_losses.append(val_loss)  

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s')  

        save_model(model, epoch+1, save_dir)  

    plot_and_save_loss(train_losses, val_losses, os.path.join(save_dir, 'loss_plot.png'))  

    return model

if __name__ == '__main__':
    # 创建模型实例  
    model = TwoStreamSNN(8, 3, 10, 8)  

    # 创建训练和验证数据加载器  
    train_dataset = TwoStreamDataset(num_samples=3)  
    val_dataset = TwoStreamDataset(num_samples=1)  
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)  

    # 训练模型  
    save_dir = 'checkpoints/'  
    trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs=2, save_dir=save_dir)