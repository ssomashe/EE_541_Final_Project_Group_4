import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SRDataset(Dataset):
    """
    A dataset class for Super Resolution (SR) tasks, inheriting from PyTorch's Dataset class.

    This class is designed to handle datasets consisting of high-resolution and corresponding low-resolution images.
    It assumes the datasets are organized in separate folders for high and low resolution images.

    Attributes:
    high_res_image_paths (list): List of file paths to high-resolution images.
    low_res_image_paths (list): List of file paths to low-resolution images.
    """

    def __init__(self, high_res_folder, low_res_folder):
        """
        Initializes the dataset object.

        Args:
        high_res_folder (str): The path to the folder containing high-resolution images.
        low_res_folder (str): The path to the folder containing low-resolution images.
        """

        # List and sort high resolution images
        high_res_list = os.listdir(high_res_folder)
        high_res_list.sort()

        # List and sort low resolution images
        low_res_list = os.listdir(low_res_folder)
        low_res_list.sort()

        # Filter out non-image files and construct full paths
        self.high_res_image_paths = [os.path.join(high_res_folder, img_name) for img_name in high_res_list if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.low_res_image_paths = [os.path.join(low_res_folder, img_name) for img_name in low_res_list if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.

        Returns:
        int: The number of high-resolution images (and equivalently, low-resolution images) in the dataset.
        """
        return len(self.high_res_image_paths)

    def __getitem__(self, index):
        """
        Retrieves a high-resolution and corresponding low-resolution image pair from the dataset by index.

        Args:
        index (int): The index of the image pair in the dataset.

        Returns:
        tuple: A tuple containing the high-resolution and low-resolution images as PyTorch tensors.
        """

        # Load images using their paths
        high_res_image_path = self.high_res_image_paths[index]
        low_res_image_path = self.low_res_image_paths[index]

        high_res_image = cv2.imread(high_res_image_path, cv2.IMREAD_COLOR)
        low_res_image = cv2.imread(low_res_image_path, cv2.IMREAD_COLOR)

        # Convert color from BGR to RGB
        high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB)
        low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)

        # Resize images to standard dimensions
        high_res_target_size = (224, 224)
        low_res_target_size = (112, 112)
        high_res_image = cv2.resize(high_res_image, high_res_target_size, interpolation=cv2.INTER_CUBIC)
        low_res_image = cv2.resize(low_res_image, low_res_target_size, interpolation=cv2.INTER_CUBIC)

        # Normalize and convert images to PyTorch tensors
        high_res_image = torch.tensor(high_res_image, dtype=torch.float32) / 255.0
        low_res_image = torch.tensor(low_res_image, dtype=torch.float32) / 255.0

        # Reorder image dimensions (HWC to CHW)
        high_res_image = high_res_image.permute(2, 0, 1)
        low_res_image = low_res_image.permute(2, 0, 1)

        return high_res_image, low_res_image

# Paths to the dataset directories
train_high_res_folder = "../data/DIV2K_train_HR"
train_low_res_folder = "../data/DIV2K_train_LR_bicubic/X2"

valid_high_res_folder = "../data/DIV2K_valid_HR"
valid_low_res_folder = "../data/DIV2K_valid_LR_bicubic/X2"

test_high_res_folder = "../data/DIV2K_test_HR"
test_low_res_folder = "../data/DIV2K_test_LR_bicubic/X2"

# Creating dataset instances
train_dataset = SRDataset(train_high_res_folder, train_low_res_folder)
valid_dataset = SRDataset(valid_high_res_folder, valid_low_res_folder)
test_dataset = SRDataset(test_high_res_folder, test_low_res_folder)

