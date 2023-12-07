import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def snr(high_res_image, output_image):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the high-resolution (HR) and output images.

    Parameters:
    high_res_image (numpy.ndarray): The high-resolution image.
    output_image (numpy.ndarray): The output image from the model.

    Returns:
    float: The SNR value.
    """
    noise = high_res_image - output_image
    return 10 * np.log10(np.mean(high_res_image ** 2) / np.mean(noise ** 2))


def ssim(high_res_image, output_image, data_range, win_size=3):
    """
    Compute the Structural Similarity Index (SSIM) between the high-resolution (HR) and output images.

    Parameters:
    high_res_image (numpy.ndarray): The high-resolution image.
    output_image (numpy.ndarray): The output image from the model.
    data_range (int): The data range of the input images (usually max - min of pixel values).
    win_size (int): The size of the window to use for SSIM computation.

    Returns:
    float: The SSIM value.
    """
    return compare_ssim(high_res_image, output_image, data_range=data_range, win_size=win_size, channel_axis=-1)


def evaluate(model, loader, device):
    """
    Evaluate the model on the given data loader using SNR, PSNR, and SSIM metrics.

    Parameters:
    model (torch.nn.Module): The neural network model to evaluate.
    loader (DataLoader): A DataLoader containing the dataset to evaluate on.
    device (torch.device): The device to run the model on.

    Returns:
    tuple: A tuple containing average SNR, PSNR, and SSIM scores for the dataset.
    """
    model.eval()
    snr_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    num_samples = 0

    with torch.no_grad():
        for high_res, low_res in loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            outputs = model(low_res)

            # Transpose the output and HR images from BCHW to BHWC format for evaluation
            outputs = outputs.squeeze().cpu().numpy().transpose(0, 2, 3, 1)
            high_res = high_res.squeeze().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(high_res.shape[0]):
                snr_value = snr(high_res[i], outputs[i])
                psnr_value = compare_psnr(high_res[i], outputs[i], data_range=high_res[i].max() - high_res[i].min())
                ssim_value = ssim(high_res[i], outputs[i], data_range=high_res[i].max() - high_res[i].min())

                snr_sum += snr_value
                psnr_sum += psnr_value
                ssim_sum += ssim_value

                num_samples += 1

    # Calculate average values
    snr_avg = snr_sum / num_samples
    psnr_avg = psnr_sum / num_samples
    ssim_avg = ssim_sum / num_samples

    return snr_avg, psnr_avg, ssim_avg
