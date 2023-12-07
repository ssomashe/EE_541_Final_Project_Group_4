import torch
from matplotlib import pyplot as plt
import cv2

def image_show(hr, lr, outputs):
    """
    Displays high-resolution (HR), low-resolution (LR), and output images side by side.

    Parameters:
    hr (numpy.ndarray): High-resolution image.
    lr (numpy.ndarray): Low-resolution image.
    outputs (numpy.ndarray): Output image from the model.

    Note: The input images are expected to be in the format (height, width, channels).
    """

    # Convert numpy arrays to torch tensors and adjust dimensions from HWC to CHW
    hr = torch.from_numpy(hr).permute(2, 0, 1)
    lr = torch.from_numpy(lr).permute(2, 0, 1)
    outputs = torch.from_numpy(outputs).permute(2, 0, 1)

    # Convert tensors back to numpy arrays in HWC format
    hr = hr.permute(1, 2, 0).numpy()
    lr = lr.permute(1, 2, 0).numpy()
    outputs = outputs.permute(1, 2, 0).numpy()

    # Plot and show the HR, LR, and output images
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    axs[0].imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    axs[0].set_title('HR')
    axs[1].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
    axs[1].set_title('LR')
    axs[2].imshow(cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')

    # Remove axis ticks for each subplot
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()  # Display the figure
