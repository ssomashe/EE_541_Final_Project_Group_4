import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

# Custom module imports
from src.data_loader import train_dataset, valid_dataset
from model.SRCNN_model import SRCNN
from utils.evaluate_metrices import evaluate
from utils.plot_srcnn import Logger, plot


def evaluate_loss(generator, discriminator, loader, device, criterion_content, criterion_adversarial):
    """
    Evaluate the losses for both generator and discriminator.

    Parameters:
    generator (torch.nn.Module): The generator model.
    discriminator (torch.nn.Module): The discriminator model.
    loader (DataLoader): DataLoader for the dataset.
    device (torch.device): Device to run the model on (CPU or GPU).
    criterion_content (Loss function): Content loss criterion.
    criterion_adversarial (Loss function): Adversarial loss criterion.

    Returns:
    Tuple[float, float]: Average generator and discriminator loss.
    """
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for high_res, low_res in loader:
            high_res = high_res.to(device)
            low_res = low_res.to(device)
            batch_size = high_res.size(0)
            total_samples += batch_size

            # Generator content loss
            fake_images = generator(low_res)
            content_loss = criterion_content(fake_images, high_res)

            # Discriminator losses
            real_output = discriminator(high_res)
            fake_output = discriminator(fake_images)
            real_loss = criterion_adversarial(real_output.squeeze(), torch.ones(batch_size, device=device))
            fake_loss = criterion_adversarial(fake_output.squeeze(), torch.zeros(batch_size, device=device))

            g_loss = content_loss  # Additional components can be included if necessary
            d_loss = real_loss + fake_loss
            
            total_g_loss += g_loss.item() * batch_size
            total_d_loss += d_loss.item() * batch_size
            
    avg_g_loss = total_g_loss / total_samples
    avg_d_loss = total_d_loss / total_samples
    return avg_g_loss, avg_d_loss


def main():
    """
    Main function to train the SRGAN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_flush_denormal(True)
    
    # Initialize Generator and Discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Training hyperparameters
    NUM_EPOCHS = 75
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_WORKERS = min(os.cpu_count(), 8)

    # Loss functions
    criterion_content = nn.MSELoss()
    criterion_adversarial = nn.BCELoss()

    # Optimizers
    optimizer_G = AdamW(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = AdamW(discriminator.parameters(), lr=LEARNING_RATE)

    # Learning rate schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min')
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min')

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Model saving directory
    MODEL_SAVE_DIR = "../output/saved_model_srgan"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_valid_psnr = 0

    # Logger for tracking training progress
    logger = Logger()

    for epoch in range(NUM_EPOCHS):
        for high_res, low_res in train_loader:
            high_res = high_res.to(device)
            low_res = low_res.to(device)
            batch_size = high_res.size(0)

            # Training the Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # Discriminator loss on real and fake images
            real_output = discriminator(high_res)
            real_loss = criterion_adversarial(real_output.squeeze(), real_labels)
            fake_images = generator(low_res)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion_adversarial(fake_output.squeeze(), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Training the Generator
            optimizer_G.zero_grad()
            content_loss = criterion_content(fake_images, high_res)
            fake_output = discriminator(fake_images)
            adversarial_loss = criterion_adversarial(fake_output.squeeze(), real_labels)
            g_loss = content_loss + 0.001 * adversarial_loss
            g_loss.backward()
            optimizer_G.step()

            # Cleanup to reduce memory usage
            del high_res, low_res, fake_images, real_output, fake_output

        # Evaluate model performance
        train_snr, train_psnr, train_ssim = evaluate(generator, train_loader, device)
        valid_snr, valid_psnr, valid_ssim = evaluate(generator, valid_loader, device)
        valid_g_loss, valid_d_loss = evaluate_loss(generator, discriminator, valid_loader, device, criterion_content, criterion_adversarial)
        
        # Log training progress
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.append(content_loss.item(), adversarial_loss.item(), train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim, timestamp)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Content Loss: {content_loss.item():.4f}, Adversarial Loss: {adversarial_loss.item():.4f}, Train SNR: {train_snr:.4f}, Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}, Valid SNR: {valid_snr:.4f}, Valid PSNR: {valid_psnr:.4f}, Valid SSIM: {valid_ssim:.4f}")

        # Save model if there is an improvement in validation PSNR
        if valid_psnr > best_valid_psnr:
            best_valid_psnr = valid_psnr
            model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_generator_{epoch + 1}.pth")
            torch.save(generator.state_dict(), model_save_path)
            print(f"Model Update: New Best PSNR: {best_valid_psnr:.4f} at Epoch {epoch + 1}")

        # Step the learning rate schedulers
        scheduler_G.step(valid_g_loss)
        scheduler_D.step(valid_d_loss)

        # Clear the GPU memory cache
        torch.cuda.empty_cache()
        gc.collect()

    # Save the training log and plot the results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"../output/train_metrics/{MODEL_NAME}_training_data_{timestamp}.csv"
    logger.save_to_csv(filename)
    plot(logger)
    
if __name__ == '__main__':
    main()
