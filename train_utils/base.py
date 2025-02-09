import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metric_history import MetricHistory


def train_model(
        model: Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        optimizer,
        criterion,
        device,
        metric_func,
        checkpoint_name: str,
        scheduler = None
    ) -> tuple[MetricHistory, MetricHistory]:
    STARTING_EPOCH = epochs // 5
    chpt_distance = max(epochs // 10, 3)
    loss_history = MetricHistory("loss", "b")
    accuracy_history = MetricHistory("accuracy", "r")
    last_saved_epoch = STARTING_EPOCH - 1
    for epoch in range(1, epochs + 1):
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_history.register_epoch_value(loss.item(), 0)
            accuracy_history.register_epoch_value(metric_func(labels, outputs), 0)
        model.eval()
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_history.register_epoch_value(0, loss.item())
                accuracy_history.register_epoch_value(0, metric_func(labels, outputs))
        loss_history.compute(len(train_loader), len(validation_loader))
        accuracy_history.compute(len(train_loader), len(validation_loader))
        if scheduler is not None:
            scheduler.step(loss_history.validation[-1])
        if epoch >= STARTING_EPOCH and epoch - chpt_distance > last_saved_epoch:
            previous_best_acc = max(accuracy_history.validation[:-2])
            val_acc = accuracy_history.validation[-1]
            if val_acc > previous_best_acc:
                _save_checkpoint(model, checkpoint_name, val_acc, epoch)
                last_saved_epoch = epoch
    val_acc = accuracy_history.validation[-1]
    _save_checkpoint(model, checkpoint_name, val_acc, epochs)
    return loss_history, accuracy_history


def _save_checkpoint(model, checkpoint_name, acc, epoch):
    acc_str = str(round(acc, 3)).replace(".", "")
    path = f"checkpoints/{checkpoint_name}_{acc_str}_epoch{epoch}.pth"
    torch.save(model.state_dict(), path)
