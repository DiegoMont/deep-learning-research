from sklearn import metrics
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from . import base
from .metric_history import MetricHistory


def classification_accuracy(targets, outputs):
    prediction = torch.argmax(outputs, dim=1)
    correct = prediction.eq(targets).sum().item()
    accuracy = correct / targets.shape[0]
    return accuracy


def print_model_performance(model: Module, dataset: Dataset, device):
    predictions = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            image, label = data[0].unsqueeze(0).to(device), data[1]
            prediction = model(image).cpu()
            predicted_class = torch.argmax(prediction, dim=1).item()
            predictions.append(predicted_class)
            ground_truth.append(label)
    print(f"Accuracy: {metrics.accuracy_score(ground_truth, predictions):.5f}")


def train_model(
        classifier: Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        optimizer,
        criterion,
        device,
        checkpoint_name: str,
        scheduler = None
    ) -> tuple[MetricHistory, MetricHistory]:
    return base.train_model(classifier, train_loader, validation_loader, epochs, optimizer,
                            criterion, device, classification_accuracy, checkpoint_name, scheduler)
