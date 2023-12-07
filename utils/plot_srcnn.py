import matplotlib.pyplot as plt
import csv

class Logger:
    """
    Logger class for tracking and recording the training progress.

    Attributes:
        loss_values (list): List to store loss values per epoch.
        train_snr_values (list): List to store training SNR values per epoch.
        valid_snr_values (list): List to store validation SNR values per epoch.
        train_psnr_values (list): List to store training PSNR values per epoch.
        valid_psnr_values (list): List to store validation PSNR values per epoch.
        train_ssim_values (list): List to store training SSIM values per epoch.
        valid_ssim_values (list): List to store validation SSIM values per epoch.
        timestamp (list): List to store timestamps for each epoch.
    """

    def __init__(self):
        """Initialize the Logger with empty lists for each metric."""
        self.loss_values = []
        self.train_snr_values = []
        self.valid_snr_values = []
        self.train_psnr_values = []
        self.valid_psnr_values = []
        self.train_ssim_values = []
        self.valid_ssim_values = []
        self.timestamp = []

    def append(self, loss, train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim, timestamp):
        """
        Append the provided metrics to their respective lists.

        Parameters:
            loss (float): Loss value for the current epoch.
            train_snr (float): Signal-to-Noise Ratio for training data.
            valid_snr (float): Signal-to-Noise Ratio for validation data.
            train_psnr (float): Peak Signal-to-Noise Ratio for training data.
            valid_psnr (float): Peak Signal-to-Noise Ratio for validation data.
            train_ssim (float): Structural Similarity Index for training data.
            valid_ssim (float): Structural Similarity Index for validation data.
            timestamp (str): Timestamp for the current epoch.
        """
        self.loss_values.append(loss)
        self.train_snr_values.append(train_snr)
        self.valid_snr_values.append(valid_snr)
        self.train_psnr_values.append(train_psnr)
        self.valid_psnr_values.append(valid_psnr)
        self.train_ssim_values.append(train_ssim)
        self.valid_ssim_values.append(valid_ssim)
        self.timestamp.append(timestamp)

    def save_to_csv(self, filename):
        """
        Save the logged metrics to a CSV file.

        Parameters:
            filename (str): The name of the file to save the data to.
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Loss', 'Train SNR', 'Valid SNR', 'Train PSNR', 'Valid PSNR', 'Train SSIM', 'Valid SSIM', 'Time Stamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.loss_values)):
                writer.writerow({
                    'Epoch': i + 1,
                    'Loss': self.loss_values[i],
                    'Train SNR': self.train_snr_values[i],
                    'Valid SNR': self.valid_snr_values[i],
                    'Train PSNR': self.train_psnr_values[i],
                    'Valid PSNR': self.valid_psnr_values[i],
                    'Train SSIM': self.train_ssim_values[i],
                    'Valid SSIM': self.valid_ssim_values[i],
                    'Time Stamp': self.timestamp[i]
                })


def plot(logger):
    """
    Plot the training progress using the data stored in the logger.

    Parameters:
        logger (Logger): An instance of the Logger class containing training metrics.
    """
    epochs = list(range(1, len(logger.loss_values) + 1))

    plt.figure(figsize=(12, 8))

    # Plotting loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, logger.loss_values, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plotting SSIM values for training and validation
    plt.subplot(2, 2, 2)
    plt.plot(epochs, logger.train_ssim_values, label="Train SSIM")
    plt.plot(epochs, logger.valid_ssim_values, label="Valid SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("Structural Similarity Index (SSIM)")
    plt.legend()

    # Plotting PSNR values for training and validation
    plt.subplot(2, 2, 3)
    plt.plot(epochs, logger.train_psnr_values, label="Train PSNR")
    plt.plot(epochs, logger.valid_psnr_values, label="Valid PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.legend()

    # Plotting SNR values for training and validation
    plt.subplot(2, 2, 4)
    plt.plot(epochs, logger.train_snr_values, label="Train SNR")
    plt.plot(epochs, logger.valid_snr_values, label="Valid SNR")
    plt.xlabel("Epoch")
    plt.ylabel("Signal-to-Noise Ratio (SNR)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../output/plots/srcnn_result_plot.png")
    plt.show()
