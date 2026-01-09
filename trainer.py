import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.data_split_utils import make_stratified_loader


class BaseTrainer(ABC):
    """Base class for training models"""
    
    def __init__(self, model, lr=0.001, batch_size=32, num_epochs=100, patience=5, save_path='model.pth', log_dir='runs'):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path
        self.log_dir = log_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    @abstractmethod
    def compute_loss(self, logits, y):
        """Compute loss. To be implemented by subclasses."""
        pass
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [train]", leave=False)
        for batch in train_iter:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.compute_loss(logits, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        return train_loss
    
    def validate_epoch(self, valid_loader):
        """Validate for one epoch"""
        self.model.eval()
        valid_loss = 0
        valid_iter = tqdm(valid_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [valid]", leave=False)
        with torch.no_grad():
            for batch in valid_iter:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                valid_loss += loss.item()
                valid_iter.set_postfix(loss=loss.item())
            valid_loss /= len(valid_loader)
        return valid_loss
    
    def train(self, train_dataset, valid_dataset):
        """Main training loop"""
        train_loader = make_stratified_loader(train_dataset, self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_valid_loss = float('inf')
        counter = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate_epoch(valid_loader)
            
            print(f'Epoch {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)
                counter = 0
            else:
                counter += 1
                if counter == self.patience:
                    break
        
        writer.close()
        return self.model


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification tasks"""
    
    def __init__(self, model, lr=0.001, batch_size=32, num_epochs=100, patience=5, save_path='model.pth', log_dir='runs'):
        super().__init__(model, lr, batch_size, num_epochs, patience, save_path, log_dir)
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, logits, y):
        """Compute cross-entropy loss"""
        return self.criterion(logits, y)


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks using CORAL"""
    
    def __init__(self, model, num_classes=6, lr=0.001, batch_size=32, num_epochs=100, patience=5, save_path='model.pth', log_dir='runs'):
        super().__init__(model, lr, batch_size, num_epochs, patience, save_path, log_dir)
        self.num_classes = num_classes
        from coral_pytorch.losses import coral_loss
        from coral_pytorch.dataset import levels_from_labelbatch
        self.coral_loss_fn = coral_loss
        self.levels_from_labelbatch = levels_from_labelbatch
    
    def compute_loss(self, logits, y):
        """Compute CORAL loss"""
        levels = self.levels_from_labelbatch(y, num_classes=self.num_classes)
        levels = levels.to(self.device)
        return self.coral_loss_fn(logits, levels)


class LabRegressionTrainer(BaseTrainer):
    """Trainer for Lab value regression tasks (3-output regression)"""
    
    def __init__(self, model, lr=0.001, batch_size=32, num_epochs=100, patience=5, save_path='model.pth', log_dir='runs'):
        super().__init__(model, lr, batch_size, num_epochs, patience, save_path, log_dir)
    
    @staticmethod
    def delta_e_1976(predictions, targets):
        """
        Compute CIE Delta E 1976 loss between predicted and target Lab values.
        
        ΔE* = √((L̂* - L*)² + (â* - a*)² + (b̂* - b*)²)
        
        Args:
            predictions: Tensor of shape (batch_size, 3) containing predicted [L*, a*, b*] values
            targets: Tensor of shape (batch_size, 3) containing target [L*, a*, b*] values
        
        Returns:
            Mean Delta E 1976 loss across the batch
        """
        diff = predictions - targets
        delta_e = torch.sqrt(torch.sum(diff ** 2, dim=1))
        return delta_e.mean()
    
    def compute_loss(self, logits, y):
        """Compute CIE Delta E 1976 loss for Lab regression"""
        return self.delta_e_1976(logits, y)
