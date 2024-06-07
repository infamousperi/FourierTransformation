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

    # Display original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')

    # Display magnitude spectrum
    axes[1].imshow(np.log1p(magnitude_spectrum), cmap='gray')
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
    fig, axes = plt.subplots(len(filtered_images), 2, figsize=(20, 20))

    for i, (threshold, filtered_img) in enumerate(filtered_images):
        # Calculate and display FFT magnitude spectrum of the filtered image
        _, filtered_magnitude_spectrum = perform_fft(filtered_img)
        axes[i, 0].imshow(np.log(1 + filtered_magnitude_spectrum), cmap='gray')
        axes[i, 0].set_title(f'Filtered Fourier Spectrum, {threshold:.5f}% filtered')

        # Display filtered image
        axes[i, 1].imshow(filtered_img, cmap='gray')
        axes[i, 1].set_title(f'Reconstructed Image (Threshold: {threshold:.5f})')

    plt.tight_layout()
    plt.show()


def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


def calculate_mse_values(original_image, filtered_images):
    mse_values = [(threshold, calculate_mse(original_image, img)) for threshold, img in filtered_images]
    return mse_values


def determine_compression_threshold(mse_values, original_image):
    max_allowed_error = 0.01 * np.mean(original_image ** 2)
    threshold_for_max_1_percent_error = next((threshold for threshold, mse in mse_values if mse <= max_allowed_error),
                                             None)
    return threshold_for_max_1_percent_error
