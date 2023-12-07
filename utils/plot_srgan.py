import matplotlib.pyplot as plt
import csv

class Logger:
    """
    A class for logging and saving training metrics.

    Attributes:
        content_loss_values (list): A list to store content loss values per epoch.
        adversarial_loss_values (list): A list to store adversarial loss values per epoch.
        train_snr_values (list): A list to store Signal-to-Noise Ratio (SNR) for training data per epoch.
        valid_snr_values (list): A list to store SNR for validation data per epoch.
        train_psnr_values (list): A list to store Peak Signal-to-Noise Ratio (PSNR) for training data per epoch.
        valid_psnr_values (list): A list to store PSNR for validation data per epoch.
        train_ssim_values (list): A list to store Structural Similarity Index (SSIM) for training data per epoch.
        valid_ssim_values (list): A list to store SSIM for validation data per epoch.
        timestamp (list): A list to store timestamps for each epoch.
    """

    def __init__(self):
        """Initializes Logger with empty lists for each metric."""
        self.content_loss_values = []
        self.adversarial_loss_values = []
        self.train_snr_values = []
        self.valid_snr_values = []
        self.train_psnr_values = []
        self.valid_psnr_values = []
        self.train_ssim_values = []
        self.valid_ssim_values = []
        self.timestamp = []

    def append(self, content_loss, adversarial_loss, train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim, timestamp):
        """
        Appends provided metrics to their respective lists.

        Parameters:
            content_loss (float): Content loss value.
            adversarial_loss (float): Adversarial loss value.
            train_snr (float): SNR for training data.
            valid_snr (float): SNR for validation data.
            train_psnr (float): PSNR for training data.
            valid_psnr (float): PSNR for validation data.
            train_ssim (float): SSIM for training data.
            valid_ssim (float): SSIM for validation data.
            timestamp (str): Timestamp for the current epoch.
        """
        self.content_loss_values.append(content_loss)
        self.adversarial_loss_values.append(adversarial_loss)
        self.train_snr_values.append(train_snr)
        self.valid_snr_values.append(valid_snr)
        self.train_psnr_values.append(train_psnr)
        self.valid_psnr_values.append(valid_psnr)
        self.train_ssim_values.append(train_ssim)
        self.valid_ssim_values.append(valid_ssim)
        self.timestamp.append(timestamp)

    def save_to_csv(self, filename):
        """
        Saves the logged metrics to a CSV file.

        Parameters:
            filename (str): The name of the file to save the data to.
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Content Loss', 'Adversarial Loss', 'Train SNR', 'Valid SNR', 'Train PSNR',
                          'Valid PSNR', 'Train SSIM', 'Valid SSIM', 'Time Stamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.content_loss_values)):
                writer.writerow({
                    'Epoch': i + 1,
                    'Content Loss': self.content_loss_values[i],
                    'Adversarial Loss': self.adversarial_loss_values[i],
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
    Plots the training and validation metrics logged in the Logger instance.

    Parameters:
        logger (Logger): An instance of the Logger class containing the training metrics.
    """
    epochs = list(range(1, len(logger.content_loss_values) + 1))

    plt.figure(figsize=(12, 8))

    # Plotting Content and Adversarial Losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, logger.content_loss_values, label="Content Loss")
    plt.plot(epochs, logger.adversarial_loss_values, label="Adversarial Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plotting SSIM for Training and Validation
    plt.subplot(2, 2, 2)
    plt.plot(epochs, logger.train_ssim_values, label="Train SSIM")
    plt.plot(epochs, logger.valid_ssim_values, label="Valid SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("Structural Similarity Index (SSIM)")
    plt.legend()

    # Plotting PSNR for Training and Validation
    plt.subplot(2, 2, 3)
    plt.plot(epochs, logger.train_psnr_values, label="Train PSNR")
    plt.plot(epochs, logger.valid_psnr_values, label="Valid PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.legend()

    # Plotting SNR for Training and Validation
    plt.subplot(2, 2, 4)
    plt.plot(epochs, logger.train_snr_values, label="Train SNR")
    plt.plot(epochs, logger.valid_snr_values, label="Valid SNR")
    plt.xlabel("Epoch")
    plt.ylabel("Signal-to-Noise Ratio (SNR)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../output/plots/srgan_result_plot.png")
    plt.show()
