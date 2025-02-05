import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.utils as vision_utils


def show_label(dataset: Dataset, index: int):
    image_arr, mask_arr = dataset[index]
    image = (image_arr * 255).type(torch.uint8)
    mask = mask_arr > 0
    colors = ["#460046", "#D2691E"]
    img = vision_utils.draw_segmentation_masks(image, mask, alpha=0.5, colors=colors)
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def pixel_accuracy(targets, outputs):
    _, predictions = torch.max(outputs.data, 1)
    correct = predictions.eq(targets).sum().item()
    accuracy = correct / torch.numel(targets)
    return accuracy


def print_accuracy_history(train_accuracy: list[float], val_accuracy: list[float]):
    epochs = range(1, len(train_accuracy) + 1)
    plt.figure(figsize=(10, 8), )
    plt.plot(epochs, train_accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b--", label="Validation accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.grid()
    plt.show()


def print_loss_history(train_loss: list[float], val_loss: list[float]):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 8), )
    plt.plot(epochs, train_loss, "ro", label="Training loss")
    plt.plot(epochs, val_loss, "r--", label="Validation loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid()
    plt.show()

