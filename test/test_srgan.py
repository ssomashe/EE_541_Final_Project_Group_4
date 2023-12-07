import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model.SRGAN_model import Generator
from src.data_loader import test_dataset
from utils.evaluate_metrices import evaluate

def main():
    """
    Main function to evaluate the Generator model (from SRGAN) on a test dataset.
    """
    # Set the device for computation
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Initialize Generator model and configure for DataParallel
    generator = Generator().to(device)
    generator = nn.DataParallel(generator)

    # Load the saved model state
    model_path = "../output/saved_model_srcnn/best_generator_69.pth"
    generator.load_state_dict(torch.load(model_path, map_location=device))

    # DataLoader setup for the test dataset
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate the model on the test dataset
    snr_avg, psnr_avg, ssim_avg = evaluate(generator, test_loader, device)
    print(f"Test: SNR: {snr_avg:.4f}, PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}")

if __name__ == "__main__":
    main()
