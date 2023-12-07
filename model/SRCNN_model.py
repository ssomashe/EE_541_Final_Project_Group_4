import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN) model.

    This model applies a series of convolutional layers to an input image
    to produce a super-resolved output.
    """

    def __init__(self):
        """
        Initializes the SRCNN model.
        """
        super(SRCNN, self).__init__()
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # First convolutional layer with ReLU activation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer with ReLU activation
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        """
        Forward pass of the SRCNN model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after applying the model.
        """
        x = self.upsample(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# Create an instance of the SRCNN model
srcnn = SRCNN()
