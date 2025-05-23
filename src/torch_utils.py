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
    criterion: callable, 
    device: Optional[torch.device] = None, 
    threshold: float = 0.5, 
    multi_class: bool = False,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
    target_col_idx: int = 1
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
        means (np.ndarray, optional): Mean values used for normalization
        stds (np.ndarray, optional): Standard deviation values used for normalization
        target_col_idx (int): Index of the target temperature column
    
    Returns:
        dict: Dictionary containing test metrics and predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    
    all_preds = []
    all_labels = []
    all_probs = []  # For storing probabilities
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if not multi_class and labels.dim() == 1 and not isinstance(criterion, torch.nn.MSELoss):
                labels = labels.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            test_total += batch_size
            
            # For multi-class classification
            if multi_class:
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                
                # Store raw outputs (logits)
                all_preds.append(outputs.cpu().numpy())
                
                # Store class indices as labels (not one-hot)
                all_labels.append(labels.cpu().numpy())
                
                # Calculate and store probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
            else:
                # Binary classification or regression
                if isinstance(criterion, torch.nn.MSELoss) or isinstance(criterion, torch.nn.L1Loss):
                    # Regression
                    all_preds.append(outputs.cpu().numpy())
                else:
                    # Binary classification
                    test_correct += ((outputs > threshold) == labels).sum().item()
                    all_preds.append(outputs.cpu().numpy())
                
                all_labels.append(labels.cpu().numpy())
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    preds_array = np.vstack(all_preds) if all_preds else np.array([])
    labels_array = np.concatenate(all_labels) if all_labels else np.array([])
    probs_array = np.vstack(all_probs) if all_probs else np.array([])
    
    # For multi-class, calculate metrics like accuracy
    if multi_class:
        # Get predicted classes
        if len(preds_array.shape) > 1 and preds_array.shape[1] > 1:
            predicted_classes = np.argmax(preds_array, axis=1)
        else:
            predicted_classes = preds_array
        
        # Calculate accuracy and confusion matrix
        from sklearn.metrics import accuracy_score, confusion_matrix
        acc = accuracy_score(labels_array, predicted_classes)
        cm = confusion_matrix(labels_array, predicted_classes)
        
        return {
            "test_loss": test_loss,
            "test_acc": acc,
            "confusion_matrix": cm,
            "predictions": preds_array,  # Raw logits
            "predicted_classes": predicted_classes,  # Class indices
            "true_labels": labels_array,  # True class indices
            "probabilities": probs_array  # Class probabilities
        }
    else:
        # For regression or binary classification
        if isinstance(criterion, torch.nn.MSELoss) or isinstance(criterion, torch.nn.L1Loss):
            # Regression metrics
            mae = np.mean(np.abs(preds_array - labels_array))
            mse = np.mean((preds_array - labels_array) ** 2)
            rmse = np.sqrt(mse)
            
            # Convert to original scale if means and stds are provided
            if means is not None and stds is not None:
                # Ensure both predictions and true values are unnormalized consistently
                preds_orig = preds_array * stds[target_col_idx] + means[target_col_idx]
                labels_orig = labels_array * stds[target_col_idx] + means[target_col_idx]
                
                # Calculate metrics in original scale
                mae_orig = np.mean(np.abs(preds_orig - labels_orig))
                mse_orig = np.mean((preds_orig - labels_orig) ** 2)
                rmse_orig = np.sqrt(mse_orig)
                
                return {
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "predictions": preds_array,  # Normalized predictions
                    "true_labels": labels_array,  # Normalized labels
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "mae_original": mae_orig,
                    "mse_original": mse_orig,
                    "rmse_original": rmse_orig,
                    "predictions_original": preds_orig,  # Unnormalized predictions
                    "true_labels_original": labels_orig  # Unnormalized labels
                }
            
            return {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "predictions": preds_array,
                "true_labels": labels_array,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            }
        else:
            # Binary classification metrics
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            
            # Convert to binary predictions for metric calculation
            binary_preds = (preds_array > threshold).astype(int)
            
            # Calculate metrics
            try:
                auc = roc_auc_score(labels_array, preds_array)
            except:
                auc = 0.0
                
            precision = precision_score(labels_array, binary_preds, zero_division=0)
            recall = recall_score(labels_array, binary_preds, zero_division=0)
            f1 = f1_score(labels_array, binary_preds, zero_division=0)
            
            return {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "predictions": preds_array,
                "true_labels": labels_array,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1
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
    class_names: Optional[List[str]] = None,
    is_regression: bool = True
) -> Dict[str, float]:
    """
    Visualize the results from the evaluate_model function.
    
    Args:
        eval_results (dict): Dictionary output from evaluate_model function
        class_names (list, optional): List of class names for classification problems
        is_regression (bool): Whether this is a regression problem
    
    Returns:
        Dict[str, float]: Dictionary containing performance metrics
    """
    import seaborn as sns
    
    if is_regression:
        # Create a 2x2 subplot for regression metrics
        fig = plt.figure(figsize=(15, 12))
        
        # Extract data
        predictions = eval_results.get("predictions_original", eval_results["predictions"])
        true_labels = eval_results.get("true_labels_original", eval_results["true_labels"])
        
        # Ensure predictions and true_labels are the same shape
        if len(predictions) != len(true_labels):
            # Handle potential shape mismatch
            min_len = min(len(predictions), len(true_labels))
            predictions = predictions[:min_len]
            true_labels = true_labels[:min_len]
        
        # Subplot 1: Key Metrics
        plt.subplot(2, 2, 1)
        metrics = {
            "MAE": eval_results.get("mae_original", eval_results["mae"]),
            "RMSE": eval_results.get("rmse_original", eval_results["rmse"]),
            "MSE": eval_results.get("mse_original", eval_results["mse"]),
            "Test Loss": eval_results["test_loss"]
        }
        
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title("Regression Performance Metrics")
        plt.xticks(rotation=45)
        
        # Subplot 2: Predicted vs Actual
        plt.subplot(2, 2, 2)
        max_samples = min(1000, len(predictions))  # Limit number of points to plot
        indices = np.random.choice(len(predictions), max_samples, replace=False)
        
        # Ensure predictions and true_labels are flattened for scatter plot
        p_flat = np.array(predictions).flatten()
        t_flat = np.array(true_labels).flatten()
        
        plt.scatter(t_flat[indices], p_flat[indices], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(t_flat), np.min(p_flat))
        max_val = max(np.max(t_flat), np.max(p_flat))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predicted vs True Values")
        
        # Subplot 3: Residuals Plot
        plt.subplot(2, 2, 3)
        residuals = p_flat - t_flat
        
        plt.scatter(t_flat[indices], residuals[indices], alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        
        # Subplot 4: Error Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color="r", linestyle="--")
        
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        
        plt.tight_layout()
        plt.show()
        
        # Create a time series plot if predictions are sequential
        plt.figure(figsize=(12, 6))
        samples_to_plot = min(200, len(predictions))
        
        plt.plot(t_flat[:samples_to_plot], label="True Values", linewidth=2)
        plt.plot(p_flat[:samples_to_plot], label="Predictions", linewidth=1, linestyle="--")
        
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("Time Series: Predicted vs True Values")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    else:
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
        
        fig = plt.figure(figsize=(15, 12))
    
        true_labels = eval_results["true_labels"]
        predictions = eval_results["predictions"]
        
        # Convert predictions to proper format for classification metrics
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class case: predictions are probabilities
            is_binary = False
            pred_classes = np.argmax(predictions, axis=1)
        else:
            is_binary = True
            predictions = np.array(predictions).flatten()
            true_labels = np.array(true_labels).flatten()
            
            if np.max(predictions) > 1 or np.min(predictions) < 0:
                predictions = 1 / (1 + np.exp(-predictions))
                
            pred_classes = (predictions > 0.5).astype(int)
            
        # Plot 1: Performance Metrics
        plt.subplot(2, 2, 1)
        metrics = {
            "Test Loss": eval_results["test_loss"],
            "Test Accuracy": eval_results["test_acc"]
        }
        
        if is_binary:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(true_labels, pred_classes)
            recall = recall_score(true_labels, pred_classes)
            f1 = f1_score(true_labels, pred_classes)
            
            metrics.update({
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
        
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title("Model Performance Metrics")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Plot 2: Confusion Matrix
        plt.subplot(2, 2, 2)
        
        if is_binary:
            cm = confusion_matrix(true_labels, pred_classes)
            labels = ["Negative", "Positive"] if class_names is None else class_names
        else:
            cm = confusion_matrix(true_labels, pred_classes)
            labels = [f"Class {i}" for i in range(cm.shape[0])] if class_names is None else class_names
            
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        # Plot 3: ROC Curve for binary, Class Accuracy for multi-class
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
            classes = np.unique(true_labels)
            class_correct = np.zeros(len(classes))
            class_total = np.zeros(len(classes))
            
            for i, c in enumerate(classes):
                class_mask = (true_labels == c)
                class_correct[i] = np.sum((pred_classes[class_mask] == c))
                class_total[i] = np.sum(class_mask)
                
            class_acc = class_correct / np.maximum(class_total, 1)  # Avoid division by zero
            
            if class_names is None:
                display_labels = [f"Class {c}" for c in classes]
            else:
                display_labels = [class_names[int(c)] if int(c) < len(class_names) else f"Class {c}" 
                                 for c in classes]
            
            sns.barplot(x=display_labels, y=class_acc)
            plt.title("Per-Class Accuracy")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
        
        # Plot 4: Precision-Recall Curve for binary, Classification Report for multi-class
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
            from sklearn.metrics import classification_report
            
            if class_names is not None and len(class_names) == len(np.unique(true_labels)):
                report = classification_report(true_labels, pred_classes, target_names=class_names)
            else:
                report = classification_report(true_labels, pred_classes)
                
            plt.axis("off")
            plt.text(0.1, 0.1, report, fontsize=10, family="monospace")
            plt.title("Classification Report")
        
        plt.tight_layout()
        plt.show()
        
    return metrics