import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vision_utils

from .metric_history import MetricHistory


def pixel_accuracy(targets, outputs):
    _, predictions = torch.max(outputs.data, 1)
    correct = predictions.eq(targets).sum().item()
    accuracy = correct / torch.numel(targets)
    return accuracy


def show_label(dataset: Dataset, index: int):
    image_arr, mask_arr = dataset[index]
    image = (image_arr * 255).type(torch.uint8)
    mask = mask_arr > 0
    colors = ["#460046", "#D2691E"]
    img = vision_utils.draw_segmentation_masks(image, mask, alpha=0.5, colors=colors) # type: ignore
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_model(
        segmenter: Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        optimizer,
        criterion,
        device,
        checkpoint_name: str,
        scheduler = None
    ) -> tuple[MetricHistory, MetricHistory]:
    STARTING_EPOCH = epochs // 5
    loss_history = MetricHistory("loss", "b")
    accuracy_history = MetricHistory("accuracy", "r")
    last_saved_epoch = -1
    for epoch in range(1, epochs + 1):
        segmenter.train()
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = segmenter(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_history.register_epoch_value(loss.item(), 0)
            accuracy_history.register_epoch_value(pixel_accuracy(labels, outputs), 0)
        segmenter.eval()
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = segmenter(images)
                loss = criterion(outputs, labels)
                loss_history.register_epoch_value(0, loss.item())
                accuracy_history.register_epoch_value(0, pixel_accuracy(labels, outputs))
        loss_history.compute(len(train_loader), len(validation_loader))
        accuracy_history.compute(len(train_loader), len(validation_loader))
        if scheduler is not None:
            scheduler.step(loss_history.validation[-1])
        if epoch >= STARTING_EPOCH and epoch - 5 > last_saved_epoch:
            previous_best_acc = max(accuracy_history.validation[:-2])
            val_acc = accuracy_history.validation[-1]
            if val_acc > previous_best_acc:
                acc = str(round(val_acc, 3)).replace(".", "")
                path = f"checkpoints/{checkpoint_name}_epoch{epoch}_{acc}.pth"
                torch.save(segmenter.state_dict(), path)
                last_saved_epoch = epoch
    return loss_history, accuracy_history
