import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional, Module
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vision_utils

from datasets.segmentation import BaseSegmentationDataset
from . import base
from .metric_history import MetricHistory


def compare_prediction(model, dataset: BaseSegmentationDataset, index: int, device):
    model.eval()
    with torch.no_grad():
        image, label = dataset[index][0].to(device), dataset[index][1].to(device)
        outputs = model(image.unsqueeze(0))
    prediction = _get_predictions(outputs)
    fig, axs = plt.subplots(1, 3, layout='tight')
    axs[0].imshow(image.permute((1, 2, 0)).cpu())
    axs[1].imshow(label.squeeze().cpu() * 127)
    axs[2].imshow(prediction.squeeze().cpu() * 127)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[2].set_axis_off()
    fig.show()


def pixel_accuracy(targets, outputs):
    predictions = _get_predictions(outputs)
    correct = predictions.eq(targets).sum().item()
    accuracy = correct / torch.numel(targets)
    return accuracy


def print_model_performace(model: Module, dataset: BaseSegmentationDataset, device):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for data in dataset:
            image, label = data[0].unsqueeze(0).to(device), data[1]
            prediction = model(image).cpu()
            accuracy += pixel_accuracy(label, prediction)
    accuracy /= len(dataset)
    print(f"Accuracy: {accuracy:.5f}")


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
    return base.train_model(segmenter, train_loader, validation_loader, epochs, optimizer,
                            criterion, device, pixel_accuracy, checkpoint_name, scheduler)


def _get_predictions(outputs: Tensor) -> Tensor:
    _, n, _, _ = outputs.shape
    if n > 1:
        predictions = torch.argmax(outputs, dim=1, keepdim=True)
    else:
        predictions = (functional.sigmoid(outputs) > 0.5).int()
    return predictions
