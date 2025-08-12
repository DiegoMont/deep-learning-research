from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional, Module
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.utils as vision_utils

from datasets.segmentation import BaseSegmentationDataset


def compare_prediction(
        model,
        dataset: BaseSegmentationDataset,
        index: int, device,
        threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        image, label = dataset[index][0].to(device), dataset[index][1].to(device)
        outputs = model(image.unsqueeze(0))
    prediction = functional.sigmoid(outputs)
    if threshold > 0:
        prediction = (prediction > threshold).float()
    fig, axs = plt.subplots(1, 3, layout='tight')
    axs[0].imshow(image.permute((1, 2, 0)).cpu())
    axs[1].imshow(label.squeeze().cpu() * 127, cmap='hot')
    axs[2].imshow(prediction.squeeze().cpu() * 127, cmap='hot')
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[2].set_axis_off()
    fig.show()


def display_conf_matrix(
        dataset: Dataset,
        model: Module,
        index: int = 0,
        threshold=0.5,
        n_samples: int = 10
):
    device = model.device
    model.eval()
    for i in range(index, index + n_samples):
        data = dataset[i]
        image, label = data[0].to(device), data[1].squeeze()
        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
        prediction = functional.sigmoid(outputs).squeeze().cpu()
        prediction = (prediction > threshold).int()
        conf_matrix = label + 2 * prediction
        red_mask = np.zeros_like(label).astype(np.uint8)
        green_mask = np.zeros_like(label).astype(np.uint8)
        blue_mask = np.zeros_like(label).astype(np.uint8)
        red_mask[conf_matrix == 1], green_mask[conf_matrix == 1], blue_mask[conf_matrix == 1] = 117, 36, 175
        red_mask[conf_matrix == 2], green_mask[conf_matrix == 2], blue_mask[conf_matrix == 2] = 255, 135, 0
        red_mask[conf_matrix == 3], green_mask[conf_matrix == 3], blue_mask[conf_matrix == 3] = 0, 182, 0
        mask_img = Image.fromarray(np.stack([red_mask, green_mask, blue_mask], axis=2), mode="RGB")
        rgb_img = v2.functional.to_pil_image(image, mode="RGB")
        overlay_img = Image.blend(rgb_img, mask_img, alpha=0.25)
        fig, axs = plt.subplots(1, 3, layout='tight')
        axs[0].imshow(rgb_img)
        axs[1].imshow(mask_img)
        axs[2].imshow(overlay_img)
        tp_patch = Patch(color='#00b600', label='TP')
        fp_patch = Patch(color='#ff8800', label='FP')
        fn_patch = Patch(color='#7524af', label='FN')
        axs[0].legend(handles=[tp_patch, fp_patch, fn_patch])
        for ax in axs: ax.set_axis_off()
        fig.show()


def pixel_accuracy(targets, outputs):
    predictions = _get_predictions(outputs)
    correct = predictions.eq(targets).sum().item()
    accuracy = correct / torch.numel(targets)
    return accuracy


def show_batch(data_loader: DataLoader, batch_id=0, n=4):
    for i, batch in enumerate(data_loader):
        if batch_id == i:
            break
    fig = plt.figure(tight_layout=True)
    for i in range(n):
        image_arr, mask_arr = batch[0][i], batch[1][i]
        image = (image_arr * 255).to(torch.uint8)
        mask = mask_arr > 0
        colors = ["#460046", "#D2691E"]
        img = vision_utils.draw_segmentation_masks(image, mask, alpha=0.5, colors=colors) # type: ignore
        npimg = img.numpy()
        ax = fig.add_subplot(1, n, i+1)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_axis_off()
        fig.show()


def show_label(dataset: Dataset, index: int):
    data = dataset[index]
    image_arr, mask_arr = data[0], data[1]
    image = (image_arr * 255).type(torch.uint8)
    mask = mask_arr > 0
    colors = ["#460046", "#D2691E"]
    img = vision_utils.draw_segmentation_masks(image, mask, alpha=0.5, colors=colors) # type: ignore
    npimg = np.array(img)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_axis_off()
    fig.show()


def _get_predictions(outputs: Tensor) -> Tensor:
    return outputs
