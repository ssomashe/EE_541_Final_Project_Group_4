import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import gc

# Import custom modules
from src.data_loader import train_dataset, valid_dataset
from model.SRCNN_model import SRCNN
from utils.evaluate_metrices import evaluate
from utils.plot_srcnn import Logger, plot


def main():
    """
    Main function to train SRCNN model.
    """

    # Set up device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure flush denormal operations
    torch.set_flush_denormal(True)

    # Initialize model and move it to the device
    srcnn = SRCNN().to(device)

    # Utilize multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        srcnn = nn.DataParallel(srcnn)

    # Define training hyperparameters
    NUM_EPOCHS = 45
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    MODEL_NAME = "srcnn"

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(srcnn.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # DataLoader setup
    NUM_WORKERS = min(os.cpu_count(), 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Directory for saving model
    MODEL_SAVE_DIR = "../output/saved_model_srcnn"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Initialize logger and best PSNR
    best_valid_psnr = 0
    logger = Logger()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        srcnn.train()

        for high_res, low_res in train_loader:
            # Transfer data to the device
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # Forward pass
            outputs = srcnn(low_res)
            loss = criterion(outputs, high_res)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Memory cleanup
            del high_res, low_res

        # Evaluate model performance on training and validation sets
        train_snr, train_psnr, train_ssim = evaluate(srcnn, train_loader, device)
        valid_snr, valid_psnr, valid_ssim = evaluate(srcnn, valid_loader, device)

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.append(loss.item(), train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim, timestamp)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, "
              f"Train SNR: {train_snr.item():.4f}, Valid SNR: {valid_snr:.4f}, "
              f"Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}, "
              f"Valid PSNR: {valid_psnr:.4f}, Valid SSIM: {valid_ssim:.4f}")

        # Save model if validation PSNR improves
        if valid_psnr > best_valid_psnr:
            best_valid_psnr = valid_psnr
            model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(srcnn.state_dict(), model_save_path)
            print(f"Model Update: New PSNR: {best_valid_psnr:.4f} at Epoch {epoch + 1}")

        # Step the scheduler and clear caches
        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

    # Save training logs and plot results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"../output/train_metrics/{MODEL_NAME}_training_data_{timestamp}.csv"
    logger.save_to_csv(filename)
    plot(logger)


if __name__ == '__main__':
    main()
