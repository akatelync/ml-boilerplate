"""
PyTorch Training Utilities
--------------------------
A collection of useful functions for training and evaluating PyTorch models.
"""

import os
import time
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import torchvision.transforms as transforms

def train_model(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: callable, 
    optimizer: torch.optim.Optimizer, 
    device: Optional[torch.device] = None,
    num_epochs: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: Optional[int] = None,
    threshold: float = 0.5,
    multi_class: bool = False,
    verbose: bool = True
) -> Tuple[dict, torch.nn.Module]:
    """
    A reusable training function for PyTorch models.
    
    Args:
        model (torch.nn.Module): PyTorch model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (callable): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str, optional): Device to train on ('cuda' or 'cpu')
        num_epochs (int): Number of epochs to train
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        early_stopping_patience (int, optional): Number of epochs to wait before early stopping
        threshold (float): Decision threshold for binary classification
        multi_class (bool): Whether this is a multi-class problem
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (history, best_model)
            - history: Dictionary containing training and validation metrics
            - best_model: Model with the best validation performance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": []
    }
    
    best_val_loss = float("inf")
    best_model_weights = None
    early_stopping_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 20)
        
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
            disable=not verbose
        )
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if not multi_class and labels.dim() == 1:
                labels = labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_total += batch_size
            
            if multi_class:
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
            else:
                train_correct += ((outputs > threshold) == labels).sum().item()
                
            train_pbar.set_postfix({
                "loss": f"{train_loss/train_total:.4f}",
                "acc": f"{train_correct/train_total:.4f}"
            })
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        val_pbar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
            disable=not verbose
        )
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if not multi_class and labels.dim() == 1:
                    labels = labels.float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_total += batch_size
                
                if multi_class:
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                else:
                    val_correct += ((outputs > threshold) == labels).sum().item()
                
                val_pbar.set_postfix({
                    "loss": f"{val_loss/val_total:.4f}",
                    "acc": f"{val_correct/val_total:.4f}"
                })
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        if verbose:
            print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
            print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
            if verbose:
                print(f"New best model (val_loss: {best_val_loss:.4f})")
        else:
            early_stopping_counter += 1
            if verbose and early_stopping_patience is not None:
                print(f"No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
        
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    time_elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    model.load_state_dict(best_model_weights)
    
    return history, model


def plot_training_history(
    history: Dict[str, List[float]], 
    figsize: Tuple[int, int] = (12, 4), 
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history (dict): Training history dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Plot accuracy
    ax2.plot(history["train_acc"], label="Training Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def evaluate_model(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    criterion: Callable, 
    device: Optional[torch.device] = None, 
    threshold: float = 0.5, 
    multi_class: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate a trained model on a test set.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        test_loader (DataLoader): DataLoader for test data
        criterion (callable): Loss function
        device (str, optional): Device to evaluate on ('cuda' or 'cpu')
        threshold (float): Decision threshold for binary classification
        multi_class (bool): Whether this is a multi-class problem
    
    Returns:
        dict: Dictionary containing test metrics (loss, accuracy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if not multi_class and labels.dim() == 1:
                labels = labels.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            total += batch_size
            
            if multi_class:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            else:
                predicted = (outputs > threshold).float()
                correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / total
    test_acc = correct / total
    
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "predictions": np.array(all_preds),
        "true_labels": np.array(all_labels)
    }


def save_model(
    model: torch.nn.Module, 
    path: str, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
    epoch: Optional[int] = None, 
    history: Optional[Dict[str, List[float]]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): PyTorch model to save
        path (str): Path to save the model
        optimizer (torch.optim.Optimizer, optional): Optimizer state
        scheduler (torch.optim.lr_scheduler, optional): Scheduler state
        epoch (int, optional): Current epoch
        history (dict, optional): Training history
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if history is not None:
        checkpoint["history"] = history
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    model: torch.nn.Module, 
    path: str, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler], Optional[int], Optional[Dict[str, List[float]]]]:
    """
    Load model checkpoint.
    
    Args:
        model (torch.nn.Module): PyTorch model to load into
        path (str): Path to the saved model
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into
        device (str, optional): Device to load the model to
    
    Returns:
        tuple: (model, optimizer, scheduler, epoch, history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", None)
    history = checkpoint.get("history", None)
    
    print(f"Model loaded from {path}")
    return model, optimizer, scheduler, epoch, history


def create_dataloaders(
    dataset: torch.utils.data.Dataset, 
    batch_size: int, 
    val_split: float = 0.2, 
    test_split: Optional[float] = None, 
    shuffle: bool = True, 
    random_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], 
          Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """
    Create train, validation, and optionally test dataloaders from a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset
        batch_size (int): Batch size for dataloaders
        val_split (float): Proportion of data to use for validation
        test_split (float, optional): Proportion of data to use for testing
        shuffle (bool): Whether to shuffle the data before splitting
        random_seed (int): Random seed for reproducibility
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory in the DataLoader
    
    Returns:
        tuple: (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    dataset_size = len(dataset)
    
    if test_split is not None:
        train_val_size = int((1 - test_split) * dataset_size)
        test_size = dataset_size - train_val_size
        
        train_val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [train_val_size, test_size]
        )
        
        train_size = int((1 - val_split) * train_val_size)
        val_size = train_val_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset, 
            [train_size, val_size]
        )
    else:
        train_size = int((1 - val_split) * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size]
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    if test_split is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.dataset)


def create_dataloaders_with_transforms(
   dataset: torch.utils.data.Dataset,
   batch_size: int,
   train_transform: transforms.Compose,
   val_transform: transforms.Compose,
   test_transform: Optional[transforms.Compose] = None,
   val_split: float = 0.2,
   test_split: Optional[float] = None,
   random_seed: int = 42,
   num_workers: int = 4,
   pin_memory: bool = True
) -> Union[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
   torch.manual_seed(random_seed)
   np.random.seed(random_seed)
   random.seed(random_seed)
   
   dataset_size = len(dataset)
   
   if test_split is not None:
       train_val_size = int((1 - test_split) * dataset_size)
       test_size = dataset_size - train_val_size
       
       train_val_dataset, test_dataset = torch.utils.data.random_split(
           dataset, 
           [train_val_size, test_size],
           generator=torch.Generator().manual_seed(random_seed)
       )
       
       train_size = int((1 - val_split) * train_val_size)
       val_size = train_val_size - train_size
       
       train_dataset, val_dataset = torch.utils.data.random_split(
           train_val_dataset, 
           [train_size, val_size],
           generator=torch.Generator().manual_seed(random_seed)
       )
   else:
       train_size = int((1 - val_split) * dataset_size)
       val_size = dataset_size - train_size
       
       train_dataset, val_dataset = torch.utils.data.random_split(
           dataset, 
           [train_size, val_size],
           generator=torch.Generator().manual_seed(random_seed)
       )
   
   train_dataset = TransformDataset(train_dataset, transform=train_transform)
   val_dataset = TransformDataset(val_dataset, transform=val_transform)
   
   train_loader = torch.utils.data.DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=num_workers,
       pin_memory=pin_memory
   )
   
   val_loader = torch.utils.data.DataLoader(
       val_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=pin_memory
   )
   
   if test_split is not None:
       test_transform_final = test_transform if test_transform else val_transform
       test_dataset = TransformDataset(test_dataset, transform=test_transform_final)
       
       test_loader = torch.utils.data.DataLoader(
           test_dataset,
           batch_size=batch_size,
           shuffle=False,
           num_workers=num_workers,
           pin_memory=pin_memory
       )
       return train_loader, val_loader, test_loader
   
   return train_loader, val_loader


def visualize_model_evaluation(
    eval_results: Dict[str, Union[float, np.ndarray]], 
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Visualize the results from the evaluate_model function.
    
    Args:
        eval_results (dict): Dictionary output from evaluate_model function
        class_names (list, optional): List of class names for classification problems
    
    Returns:
        Dict[str, float]: Dictionary containing performance metrics
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
    
    fig = plt.figure(figsize=(15, 12))
    
    true_labels = eval_results["true_labels"]
    predictions = eval_results["predictions"]
    
    is_binary = (len(predictions.shape) == 1 or predictions.shape[1] == 1)
    
    plt.subplot(2, 2, 1)
    metrics = {
        "Test Loss": eval_results["test_loss"],
        "Test Accuracy": eval_results["test_acc"]
    }
    
    if is_binary:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        if np.max(predictions) > 1:
            y_pred = (predictions > 0.5).astype(int)
        else:
            y_pred = predictions
            
        precision = precision_score(true_labels, y_pred)
        recall = recall_score(true_labels, y_pred)
        f1 = f1_score(true_labels, y_pred)
        
        metrics.update({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
    
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Model Performance Metrics")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    
    if is_binary:
        binary_preds = (predictions > 0.5).astype(int)
        cm = confusion_matrix(true_labels, binary_preds)
        labels = ["Negative", "Positive"] if class_names is None else class_names
    else:
        cm = confusion_matrix(true_labels, predictions.argmax(axis=1))
        labels = [f"Class {i}" for i in range(cm.shape[0])] if class_names is None else class_names
        
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.subplot(2, 2, 3)
    
    if is_binary:
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
    else:
        class_correct = np.zeros(len(labels))
        class_total = np.zeros(len(labels))
        
        pred_classes = predictions.argmax(axis=1)
        for i in range(len(true_labels)):
            label = int(true_labels[i])
            class_correct[label] += (pred_classes[i] == label)
            class_total[label] += 1
            
        class_acc = class_correct / class_total
        
        sns.barplot(x=labels, y=class_acc)
        plt.title("Per-Class Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    
    if is_binary:
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
    else:
        report = classification_report(true_labels, predictions.argmax(axis=1), target_names=labels)
        plt.axis("off")
        plt.text(0.1, 0.1, report, fontsize=10, family="monospace")
        plt.title("Classification Report")
    
    plt.tight_layout()
    plt.show()
    
    return metrics