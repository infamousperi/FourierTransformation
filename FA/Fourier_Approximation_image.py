import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.array(img)


def perform_fft(image_array):
    fft_img = fft2(image_array)
    fft_img_shifted = fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_img_shifted)
    return fft_img_shifted, magnitude_spectrum


def plot_magnitude_spectrum(original_image, magnitude_spectrum, title):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Set white background for each subplot
    for ax in axes:
        ax.set_facecolor('white')

    # Display original image
    axes[0].imshow(original_image, cmap='inferno')
    axes[0].set_title('Original Image')

    # Display magnitude spectrum
    axes[1].imshow(np.log1p(magnitude_spectrum), cmap='inferno')
    axes[1].set_title(title)

    plt.show()


def filter_fft_components(fft_img_shifted, magnitude_spectrum, thresholds):
    max_amplitude = np.max(magnitude_spectrum)
    filtered_images = []
    for threshold in thresholds:
        threshold_value = (threshold / 100) * max_amplitude
        filtered_fft = np.where(magnitude_spectrum > threshold_value, fft_img_shifted, 0)
        filtered_img = np.abs(ifft2(ifftshift(filtered_fft)))
        filtered_images.append((threshold, filtered_img))
    return filtered_images


def plot_filtered_images(original_image, filtered_images):
    n = len(filtered_images)
    fig, axes = plt.subplots(n, 2, figsize=(20, 10 * n))

    if n == 1:
        axes = [axes]  # Wrap axes in a list to make indexing consistent

    for i, (threshold, filtered_img) in enumerate(filtered_images):
        # Calculate and display FFT magnitude spectrum of the filtered image
        _, filtered_magnitude_spectrum = perform_fft(filtered_img)
        axes[i][0].imshow(np.log(1 + filtered_magnitude_spectrum), cmap='inferno')
        axes[i][0].set_title(f'Filtered Fourier Spectrum, {threshold:.5f}% filtered')

        # Display filtered image
        axes[i][1].imshow(filtered_img, cmap='inferno')
        axes[i][1].set_title(f'Reconstructed Image (Threshold: {threshold:.5f}%)')

    plt.tight_layout()
    plt.show()


def compute_mse(original_image, reconstructed_image):
    return np.mean((original_image - reconstructed_image) ** 2)


def find_max_compression_threshold(original_image, fft_img_shifted, magnitude_spectrum, max_error_percentage):
    max_amplitude = np.max(magnitude_spectrum)
    thresholds = np.arange(0, 100, 0.00001)  # Check every 0.1% from 0% to 99.9%
    for threshold in thresholds:
        threshold_value = (threshold / 100) * max_amplitude
        filtered_fft = np.where(magnitude_spectrum > threshold_value, fft_img_shifted, 0)
        filtered_img = np.abs(ifft2(ifftshift(filtered_fft)))
        mse = compute_mse(original_image, filtered_img)
        error_percentage = (mse / np.mean(original_image ** 2)) * 100
        print(f'Threshold: {threshold}%, Error: {error_percentage:.4f}%')
        if error_percentage > max_error_percentage:
            return threshold - 0.00001  # Return the previous threshold that was within the limit
    return thresholds[-1]  # If all thresholds are within the limit, return the last one