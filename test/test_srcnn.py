import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model.SRCNN_model import SRCNN
from src.data_loader import test_dataset
from utils.evaluate_metrices import evaluate

def main():
    """
    Main function to evaluate the SRCNN model on a test dataset.
    """
    # Set the device for computation
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Initialize SRCNN model and configure for DataParallel
    srcnn = SRCNN().to(device)
    srcnn = nn.DataParallel(srcnn)

    # Load the saved model state
    model_path = "../output/saved_model_srcnn/best_model_epoch_63.pth"
    srcnn.load_state_dict(torch.load(model_path, map_location=device))

    # DataLoader setup for the test dataset
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate the model on the test dataset
    snr_avg, psnr_avg, ssim_avg = evaluate(srcnn, test_loader, device)
    print(f"Test: SNR: {snr_avg}, PSNR: {psnr_avg}, SSIM: {ssim_avg}")

if __name__ == "__main__":
    main()
