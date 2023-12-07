import torch
import torch.nn as nn

class Block(nn.Module):
    """
    A basic building block for the generator with convolutional layers,
    batch normalization, and PReLU activation.
    """

    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1):
        """
        Initializes the Block.

        Parameters:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=padding),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x0):
        """
        Forward pass of the block.

        Parameters:
        x0 (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after applying the block.
        """
        x1 = self.layer(x0)
        return x0 + x1

class Generator(nn.Module):
    """
    Generator model for super-resolution tasks, using residual blocks.
    """

    def __init__(self, scale=2):
        """
        Initializes the Generator model.

        Parameters:
        scale (int): Scale factor for upsampling.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_block = nn.Sequential(
            Block(),
            Block(),
            Block(),
            Block(),
            Block(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )
        self.conv4 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        """
        Forward pass of the generator.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after applying the generator.
        """
        x0 = self.conv1(x)
        x = self.residual_block(x0)
        x = self.conv2(x)
        x = self.conv3(x + x0)
        x = self.conv4(x)
        return x

class DownSample(nn.Module):
    """
    Downsampling block using a convolutional layer, batch normalization,
    and LeakyReLU activation.
    """

    def __init__(self, input_channel, output_channel, stride, kernel_size=3, padding=1):
        """
        Initializes the DownSample block.

        Parameters:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        stride (int): Stride of the convolution.
        kernel_size (int): Size of the kernel.
        padding (int): Padding added to both sides of the input.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the downsampling block.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after applying the block.
        """
        x = self.layer(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator model for GAN-based super-resolution tasks.
    """

    def __init__(self):
        """
        Initializes the Discriminator model.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DownSample(64, 64, stride=2, padding=1),
            DownSample(64, 128, stride=1, padding=1),
            DownSample(128, 128, stride=2, padding=1),
            DownSample(128, 256, stride=1, padding=1),
            DownSample(256, 256, stride=2, padding=1),
            DownSample(256, 512, stride=1, padding=1),
            DownSample(512, 512, stride=2, padding=1),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after applying the discriminator.
        """
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x

# Test the models
if __name__ == '__main__':
    g = Generator()
    a = torch.rand([1, 3, 64, 64])
    d = Discriminator()
    b = torch.rand([2, 3, 512, 512])
    