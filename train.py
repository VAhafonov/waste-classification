"""
Training Script for Waste Classification
Includes TensorBoard logging and YAML config loading
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from tqdm import tqdm
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from models import create_model
from dataset_v1 import create_data_loaders


def worker_init_fn(worker_id):
    """Initialize worker seeds for DataLoader reproducibility"""
    # Get the base seed from the current random state
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)                                                                                                                                   
    torch.cuda.manual_seed(worker_seed)                                                                                                                              
    torch.cuda.manual_seed_all(worker_seed)                                                                                          
    np.random.seed(worker_seed)                                                                                                             
    random.seed(worker_seed)                                                                                                       


def seed_everything(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


class ModelTrainer:
    """Helper class for training the model"""
    
    def __init__(self, model, num_classes, device='cpu'):
        self.model = model.to(device)
        self.device = device
        # Initialize accuracy metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes).to(device)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes).to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        self.train_accuracy.reset()
        self.train_f1.reset()
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
           
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            # print(f" batch {batch_idx} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Update accuracy metric
            self.train_accuracy.update(output, target)
            self.train_f1.update(output, target)
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = self.train_accuracy.compute().item() * 100.0
        f1 = self.train_f1.compute().item()
        return avg_loss, accuracy, f1
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        self.val_accuracy.reset()
        self.val_f1.reset()
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                # Update accuracy metric
                self.val_accuracy.update(output, target)
                self.val_f1.update(output, target)
        avg_loss = total_loss / len(dataloader)
        accuracy = self.val_accuracy.compute().item() * 100.0
        f1 = self.val_f1.compute().item()
        return avg_loss, accuracy, f1


def save_model(model, filepath, epoch, optimizer, loss, additional_info=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")


def load_model(config, filepath, device='cpu'):
    """Load model from checkpoint"""
    try:
        model = create_model(config['model'])
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        
        print(f"Model loaded from {filepath}")
        if epoch is not None:
            print(f"Loaded from epoch {epoch} with loss {loss:.4f}")
        
        return model, epoch, loss
        
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None, None, None


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = float(config['training']['weight_decay'])
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training']['momentum']
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    print(f"Created {optimizer_name.upper()} optimizer with lr={lr}")
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    scheduler_type = config['scheduler']['type'].lower()
    
    if scheduler_type == 'step':
        step_size = config['scheduler']['step_size']
        gamma = config['scheduler']['gamma']
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Created StepLR scheduler with step_size={step_size}, gamma={gamma}")
    elif scheduler_type == 'cosine':
        T_max = config['training']['epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        print(f"Created CosineAnnealingLR scheduler with T_max={T_max}")
    elif scheduler_type == 'none':
        scheduler = None
        print("No scheduler will be used")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return scheduler


def setup_logging(config, config_filename):
    """Setup TensorBoard logging"""
    if not config['logging']['tensorboard']:
        return None
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_filename = os.path.basename(config_filename).split('.')[0] + '_' + timestamp
    log_dir = os.path.join(config['logging']['log_dir'], log_dir_filename)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging setup at: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={config['logging']['log_dir']}")
    
    return writer


def log_model_info(writer, model, config):
    """Log model information to TensorBoard"""
    if writer is None:
        return
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log model info as text
    info_text = f"""
    Model: ResNet34
    Classes: {config['model']['num_classes']}
    Pretrained: {config['model']['pretrained']}
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Random Seed: {config['training'].get('seed', 42)}
    """
    
    writer.add_text("Model Info", info_text, 0)
    
    # Log config as text
    config_text = yaml.dump(config, default_flow_style=False)
    writer.add_text("Configuration", f"```yaml\n{config_text}\n```", 0)


def train_model(config, model, train_loader, val_loader, device, config_filename):
    """Main training function"""
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    trainer = ModelTrainer(model, config['model']['num_classes'], device)
    
    # Setup logging
    writer = setup_logging(config, config_filename)
    log_model_info(writer, model, config)
    
    # Training parameters
    epochs = config['training']['epochs']
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    save_dir = os.path.join(config['logging']['save_dir'], os.path.basename(config_filename).split('.')[0])
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training variables
    best_val_accuracy = 0.0
    start_time = time.time()
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_accuracy, train_f1 = trainer.train_epoch(train_loader, optimizer, criterion, scheduler)
        
        # Validation phase
        val_loss, val_accuracy, val_f1 = trainer.validate(val_loader, criterion)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log to console
        print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}")
        
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            writer.add_scalar('F1/Train', train_f1, epoch)
            writer.add_scalar('F1/Validation', val_f1, epoch)
            if scheduler:
                writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_model(model, checkpoint_path, epoch, optimizer, val_loss, {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'config': config
            })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(save_dir, "best_model.pth")
            save_model(model, best_model_path, epoch, optimizer, val_loss, {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'config': config
            })
            print(f"  New best model saved! Val Acc: {val_accuracy:.2f}%")
        
        print("-" * 50)
    
    # Training completed
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Final model save
    final_model_path = os.path.join(save_dir, "final_model.pth")
    save_model(model, final_model_path, epochs-1, optimizer, val_loss, {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'config': config,
        'total_training_time': total_time
    })
    
    if writer:
        writer.close()
    
    return model, best_val_accuracy


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Waste Classification Model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cpu, cuda, or auto')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return
    
    # Set random seed for reproducibility
    seed = config['training'].get('seed', 42)  # Default to 42 if not specified
    seed_everything(seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config['dataset'], worker_init_fn)
    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Validation dataset: {len(val_loader.dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = create_model(config['model'])
    
    # Check for checkpoint resuming (weights only)
    checkpoint_config = config.get('checkpoint', {})
    if checkpoint_config.get('resume', False):
        checkpoint_path = checkpoint_config.get('path', '')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model weights from checkpoint: {checkpoint_path}")
            loaded_model, epoch, loss = load_model(config, checkpoint_path, device)
            if loaded_model is not None:
                model = loaded_model
                print("Model weights loaded successfully!")
            else:
                print("Failed to load checkpoint. Starting from scratch.")
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet34")
    print(f"Parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    # Train model
    trained_model, best_accuracy = train_model(config, model, train_loader, val_loader, device, args.config)
    
    print("\nTraining completed successfully!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
