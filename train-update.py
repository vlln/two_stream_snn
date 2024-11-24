from dataset import TwoStreamDataset
from model import TwoStreamSNN
from torch.utils.data import DataLoader
import torch
from torch import nn
import os  
import time  
import matplotlib.pyplot as plt  
import torch.optim as optim  
from tqdm import tqdm

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

def plot_and_save_acc(train_losses, val_losses, save_path):  
    plt.figure(figsize=(8, 6))  
    plt.plot(train_losses, label='Training Accuracy')  
    plt.plot(val_losses, label='Validation Accuracy')  
    plt.xlabel('Epoch')  
    plt.ylabel('Acc')  
    plt.title('Training and Validation Acc')  
    plt.legend()  
    plt.savefig(save_path)  
    plt.close()  


def train_model(model, train_dataloader, val_dataloader, num_epochs, save_dir, checkpoint_path):  
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'  
    model.to(device)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    criterion = nn.CrossEntropyLoss()  

    # 初始化损失和准确率列表
    train_losses = []  
    val_losses = []  
    train_accuracies = []  
    val_accuracies = []  
    best_val_accuracy = 0.0

    start_epoch = 0  # 用于断点续训

    # 如果指定了检查点路径，则加载检查点
    # if checkpoint_path and os.path.exists(checkpoint_path):  
    #     print(f"Loading checkpoint from {checkpoint_path}...")
    #     checkpoint = torch.load(checkpoint_path)  
    #     model.load_state_dict(checkpoint)  
    #     print(f"Resumed training from epoch {start_epoch+1}.")

    for epoch in tqdm(range(start_epoch, num_epochs)):  
        start_time = time.time()  

        # 训练模式  
        model.train()  
        train_loss = 0.0  
        correct_train = 0  
        total_train = 0  

        for batch in train_dataloader:  
            rgb_data, flow_data, labels = batch['rgb'].to(device), batch['flow'].to(device), batch['label'].to(device)  
            optimizer.zero_grad()  
            outputs = model(rgb_data, flow_data)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            train_loss += loss.item()  

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(train_dataloader)  
        train_losses.append(train_loss)  
        train_accuracy = correct_train / total_train  
        train_accuracies.append(train_accuracy)

        # 验证模式  
        model.eval()  
        val_loss = 0.0  
        correct_val = 0  
        total_val = 0  

        with torch.no_grad():  
            for batch in val_dataloader:  
                rgb_data, flow_data, labels = batch['rgb'].to(device), batch['flow'].to(device), batch['label'].to(device)  
                outputs = model(rgb_data, flow_data)  
                loss = criterion(outputs, labels)  
                val_loss += loss.item()  

                # 计算准确率
                _, predicted = torch.max(outputs, 1)  
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_dataloader)  
        val_losses.append(val_loss)  
        val_accuracy = correct_val / total_val  
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val Accuracy: {best_val_accuracy:.4f}")


        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, '
              f'Time: {time.time() - start_time:.2f}s') 
        
        current_model_path = os.path.join(save_dir, f'model_current_epoch.pth')
        torch.save(model.state_dict(), current_model_path)

    with open(os.path.join(save_dir, 'logs.txt'), 'w') as f:
        f.write("Epoch, Train Loss, Train Acc, Val Loss, Val Acc\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1}, {train_losses[i]:.4f}, {train_accuracies[i]:.4f}, {val_losses[i]:.4f}, {val_accuracies[i]:.4f}\n")
  

    plot_and_save_loss(train_losses, val_losses, os.path.join(save_dir, 'loss_plot.png'))  
    plot_and_save_acc(train_accuracies, val_accuracies, os.path.join(save_dir, 'acc_plot.png')) 

    return train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == '__main__':
    # 创建模型实例  
    flow_channels = 2
    rgb_channels = 3
    num_classes = 101
    time_step = 4
    model = TwoStreamSNN(flow_channels, rgb_channels, num_classes, time_step)

    flow_path = '/data1/home/sunyi/large-files/Python_Objects/spiking_neural_network/UCF101-flow-data.pth'
    rgb_path = '/data1/home/sunyi/large-files/Python_Objects/spiking_neural_network/UCF-101-RGB-data.pth'

    dataset = TwoStreamDataset(flow_path, rgb_path) 
    # 创建训练和验证数据加载器  
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)  
    checkpoint_path = '/data1/home/sunyi/large-files/Python_Objects/spiking_neural_network/checkpoints/model_epoch_50.pth'

    # 训练模型  
    save_dir = './checkpoints/'  
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, 
                                                                             train_dataloader, 
                                                                             val_dataloader, 
                                                                             num_epochs=1000, 
                                                                             save_dir=save_dir,
                                                                             checkpoint_path=checkpoint_path
                                                                            )

